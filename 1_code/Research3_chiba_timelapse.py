#!/Users/seimiyaakirayuu/Desktop/project/.venv/bin/python3
import ee

# GEEの認証
# ee.Authenticate()
ee.Initialize()

# 観測地点の設定（Web Mercator投影を考慮した正方形ポリゴン）
# 元のポリゴンの中心点と範囲を計算
original_min_lon = 140.048947
original_max_lon = 140.068946
original_min_lat = 35.14381
original_max_lat = 35.161424

center_lon = (original_min_lon + original_max_lon) / 2
center_lat = (original_min_lat + original_max_lat) / 2

# 元の範囲を計算
lon_range = original_max_lon - original_min_lon
lat_range = original_max_lat - original_min_lat

# Web Mercator投影を考慮した補正
# 緯度のコサインで補正（高緯度ほど経度の距離が短くなる）
import math
lat_correction = math.cos(math.radians(center_lat))

# 実際の距離で正方形にするため、緯度範囲を基準に経度範囲を調整
# 緯度範囲に対応する経度範囲 = 緯度範囲 / 補正係数
effective_lat_range = lat_range
effective_lon_range = effective_lat_range / lat_correction

# 正方形のサイズ（実際の距離で同じになるように）
square_lat_size = effective_lat_range
square_lon_size = effective_lon_range

# 正方形の座標を計算（中心を基準に）
top_lat = center_lat + square_lat_size / 2
bottom_lat = center_lat - square_lat_size / 2
left_lon = center_lon - square_lon_size / 2
right_lon = center_lon + square_lon_size / 2

# 正方形のポリゴン座標
geometry = ee.Geometry.Polygon(
    [
        [left_lon, bottom_lat],
        [left_lon, top_lat],
        [right_lon, top_lat],
        [right_lon, bottom_lat],
        [left_lon, bottom_lat]
    ]
)

print(f"正方形ポリゴン（Web Mercator補正後）:")
print(f"  緯度範囲: {square_lat_size:.6f}度")
print(f"  経度範囲: {square_lon_size:.6f}度 (補正係数: {lat_correction:.4f})")
print(f"  座標範囲: 経度 [{left_lon:.6f}, {right_lon:.6f}], 緯度 [{bottom_lat:.6f}, {top_lat:.6f}]")

# Sentinel-2 ImageCollectionの選択
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

# 対象日付のリスト（Research_chiba2.pyと同じ日程）
target_dates = [
    "2025-04-21",
    "2025-08-24",
    "2025-09-08",
    "2025-10-28",
    "2025-11-07"
]

# 各日付の画像を取得して結合（前後3日間の範囲で取得）
from datetime import datetime, timedelta

collections = []
for date in target_dates:
    # 前後3日間の範囲で画像を取得（Research_chiba2.pyと同じ方法）
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")
    
    # 前後3日間の範囲で画像を取得
    date_collection_range = (
        collection
        .filterDate(f"{start_date}T00:00:00", f"{end_date}T23:59:59")
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .sort("system:time_start")
    )
    
    # 取得できた画像数を確認
    try:
        count_range = date_collection_range.size().getInfo()
        
        if count_range > 0:
            # 対象日付に最も近い1枚を取得
            target_time = ee.Date(f"{date}T12:00:00").millis()
            date_collection = date_collection_range.map(
                lambda img: img.set("time_diff", ee.Number(img.get("system:time_start")).subtract(target_time).abs())
            ).sort("time_diff").limit(1)
            collections.append(date_collection)
        else:
            continue
        
    except Exception as e:
        continue

# すべてのコレクションを結合
if len(collections) == 0:
    raise ValueError("取得できた画像がありません。")
elif len(collections) == 1:
    collection = collections[0]
else:
    collection = ee.ImageCollection(collections[0])
    for col in collections[1:]:
        collection = collection.merge(col)

# 時系列順にソート
collection = collection.sort("system:time_start")

# 雲マスク処理関数
def maskS2clouds(image):
    qa = image.select("QA60")
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    # Surface Reflectanceは10000倍されているので、10000で割る
    masked_image = image.updateMask(mask).divide(10000)
    
    # ポリゴン範囲内の平均値を計算（黒塗りや欠損画像を検出するため）
    # 平均値が非常に低い（0.01以下）場合は黒塗りと判断
    stats = masked_image.select("B4").reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    )
    mean_value = ee.Number(stats.get("B4"))
    
    # 有効ピクセル数の割合も計算
    valid_pixels = masked_image.select("B4").mask().reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    )
    total_pixels = ee.Image.constant(1).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    )
    valid_ratio = ee.Number(valid_pixels.get("B4")).divide(ee.Number(total_pixels.get("constant")))
    
    return masked_image.copyProperties(image, ["system:time_start"]).set({
        "valid_ratio": valid_ratio,
        "mean_B4": mean_value
    })

# 雲マスク処理を適用
collection = collection.map(maskS2clouds)

# フィルタリング：有効ピクセル数50%以上、かつ平均値0.01以上（黒塗りを除外）
collection = collection.filter(
    ee.Filter.And(
        ee.Filter.gte("valid_ratio", 0.5),
        ee.Filter.gte("mean_B4", 0.01)
    )
)

# 表示させるバンドの選択（RGB: B4=Red, B3=Green, B2=Blue）
# 動画エクスポートではクリップ処理は不要（regionパラメータで自動的に範囲が指定される）
bands = collection.select(["B4", "B3", "B2"])

# データの存在確認
image_count = collection.size().getInfo()
print(f"最終的な取得可能な画像数: {image_count}")

if image_count == 0:
    print("警告: 指定された期間・範囲に画像が存在しません。")
    print("日付範囲またはポリゴン範囲を確認してください。")
elif image_count < 5:
    print(f"警告: 画像数が少ないです（{image_count}枚）。動画が短くなる可能性があります。")

# 画像データを8 bitに変換する関数（明るさとコントラストを向上）
# Sentinel-2 Surface Reflectanceは0-1の範囲（10000で割った後）なので、255倍で8bitに変換
def convertBit(image):
    # 明るさとコントラストを向上させる処理
    # 1. コントラスト調整：値を0.02-0.98の範囲に正規化（極端な値を除外）
    normalized = image.clamp(0.02, 0.98).subtract(0.02).divide(0.96)
    
    # 2. ガンマ補正で明るさを向上（ガンマ=0.7で明るくする）
    gamma = 0.7
    brightened = normalized.pow(gamma)
    
    # 3. さらに明るさを調整（1.2倍）
    brightened = brightened.multiply(1.2).clamp(0, 1)
    
    # 4. 0-255の範囲にスケール
    return brightened.multiply(255).uint8()

# 画像データの8 bit化
output = bands.map(convertBit)

# ポリゴンのアスペクト比を計算（Web Mercator投影を考慮）
# ポリゴンのバウンディングボックスを取得
bbox = geometry.bounds()
coords = bbox.coordinates().get(0).getInfo()
min_lon, min_lat = coords[0]
max_lon, max_lat = coords[2]

# 経度と緯度の範囲を計算
lon_range = max_lon - min_lon
lat_range = max_lat - min_lat

# Web Mercator投影では、緯度によって経度の距離が変わる
# 中心緯度での補正を計算
center_lat = (min_lat + max_lat) / 2
# 緯度のコサインで補正（高緯度ほど経度の距離が短くなる）
lat_correction = abs(1.0 / (abs(center_lat) * 3.14159 / 180.0) if center_lat != 0 else 1.0)
# より正確な補正：緯度のコサインを使用
import math
lat_correction = math.cos(math.radians(center_lat))

# 補正後のアスペクト比を計算
aspect_ratio = (lon_range * lat_correction) / lat_range

# 動画の高さを設定（720ピクセル）
video_height = 720
# 幅をアスペクト比に基づいて計算
video_width = int(video_height * aspect_ratio)

print(f"ポリゴン範囲: 経度 {lon_range:.6f}度, 緯度 {lat_range:.6f}度")
print(f"中心緯度: {center_lat:.6f}度, 補正係数: {lat_correction:.4f}")
print(f"動画サイズ: {video_width}x{video_height} (アスペクト比: {aspect_ratio:.2f})")

# 動画データとして出力する条件の設定
# dimensionsを単一の数値にすることで、Earth Engineが自動的にアスペクト比を計算
output_viedo = ee.batch.Export.video.toDrive(
    output,
    description="Chiba_NDVI_timelapse",
    dimensions=720,  # 最大サイズ（高さまたは幅の大きい方）
    framesPerSecond=2,
    region=geometry,
    maxFrames=10000,
    crs='EPSG:3857',  # Web Mercator投影を指定
)

# 動画データ出力の実行
output_viedo.start()
print("動画のエクスポートタスクを開始しました。Google Driveで確認してください。")
print(f"タスクID: {output_viedo.id}")
print(f"タスク状態: {output_viedo.state}")

# タスクの状態を確認（数秒待ってから）
import time
time.sleep(3)
task_list = ee.batch.Task.list()
print(f"\nアクティブなタスク数: {len([t for t in task_list if t.state in ['READY', 'RUNNING']])}")
for task in task_list[:5]:  # 最新5件を表示
    if 'Chiba_NDVI_timelapse' in task.config.get('description', ''):
        print(f"  タスク: {task.config.get('description', 'N/A')}, 状態: {task.state}, ID: {task.id}")
