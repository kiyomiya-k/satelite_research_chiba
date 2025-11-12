#!/Users/seimiyaakirayuu/Desktop/project/.venv/bin/python3
"""
NDVI時系列変化を計算し、グラフとマップを生成するスクリプト
Research_chiba_1.ipynbの日付に対応
"""

import altair as alt
import colorcet as cc
import ee
import folium
import pandas as pd
from altair_saver import save
from datetime import datetime, timedelta
import io
import os
import time
import shutil
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ============================================================================
# 初期化と設定
# ============================================================================

# Google Earth Engineの初期化
ee.Initialize()

# ポリゴン座標の定義（[経度, 緯度]の形式）
coords = [
    [140.048947, 35.14381],
    [140.048947, 35.161424],
    [140.068946, 35.161424],
    [140.068946, 35.14381],
    [140.048947, 35.14381]
]
stat_region = ee.Geometry.Polygon(coords)

# ポリゴンの中心点を計算
center_lon = (140.048947 + 140.068946) / 2
center_lat = (35.14381 + 35.161424) / 2

# ポリゴンのサイズを計算（経度と緯度の範囲）
lon_range = 140.068946 - 140.048947
lat_range = 35.161424 - 35.14381

# 1/4サイズの正方形を作成（小さい方の範囲を使用）
quarter_size = min(lon_range, lat_range) / 4
quarter_square_coords = [
    [center_lon - quarter_size, center_lat - quarter_size],
    [center_lon - quarter_size, center_lat + quarter_size],
    [center_lon + quarter_size, center_lat + quarter_size],
    [center_lon + quarter_size, center_lat - quarter_size],
    [center_lon - quarter_size, center_lat - quarter_size]
]
stat_region_quarter = ee.Geometry.Polygon(quarter_square_coords)

# 対象日付のリスト
target_dates = [
    "2025-04-21",
    "2025-08-24",
    "2025-09-08",
    "2025-10-28",
    "2025-11-07"
]

# ============================================================================
# 関数定義
# ============================================================================

def add_ee_layer(self, ee_image_object, vis_params, name):
    """
    FoliumマップにGoogle Earth Engineの画像レイヤーを追加する関数
    
    Args:
        self: Foliumマップオブジェクト
        ee_image_object: Earth Engine画像オブジェクト
        vis_params: 可視化パラメータ（min, max, bands, paletteなど）
        name: レイヤー名
    """
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True,
    ).add_to(self)


def maskS2clouds(image):
    """
    Sentinel-2画像から雲をマスクする関数
    
    Args:
        image: Sentinel-2画像
        
    Returns:
        雲マスクを適用し、10000で割った画像（Surface Reflectanceを0-1の範囲に変換）
    """
    qa = image.select("QA60")
    cloudBitMask = 1 << 10  # ビット10: 雲
    cirrusBitMask = 1 << 11  # ビット11: 巻雲
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])


def reduce_region_function(img):
    """
    画像からNDVI統計を計算する関数
    ポリゴン全体、中心点、1/4正方形の3つの領域で統計を計算
    
    Args:
        img: NDVI画像
        
    Returns:
        統計値を含むFeatureオブジェクト
    """
    ndvi_band = img.select("NDVI")
    center_point = stat_region.centroid()
    
    # ポリゴン全体の統計（平均、最小、最大、標準偏差）
    region_stat = ndvi_band.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=stat_region, 
        scale=10,  # Sentinel-2の解像度（10m）
        maxPixels=1e9
    )
    
    # 中心点のNDVI値
    center_stat = ndvi_band.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=center_point,
        scale=10,
        maxPixels=1e9
    )
    
    # 1/4サイズの正方形内の平均NDVI値
    quarter_square_stat = ndvi_band.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=stat_region_quarter,
        scale=10,
        maxPixels=1e9
    )
    
    # タイムスタンプを取得
    time_start = img.get("system:time_start")
    
    # 統計値の取得（nullチェック付き）
    ndvi_mean = region_stat.get("NDVI_mean")
    ndvi_center = center_stat.get("NDVI")
    ndvi_quarter = quarter_square_stat.get("NDVI")
    
    # nullの場合は0をデフォルト値として設定
    stat = region_stat.set("NDVI", ee.Algorithms.If(
        ee.Algorithms.IsEqual(ndvi_mean, None),
        ee.Number(0),
        ndvi_mean
    ))
    stat = stat.set("NDVI_center", ee.Algorithms.If(
        ee.Algorithms.IsEqual(ndvi_center, None),
        ee.Number(0),
        ndvi_center
    ))
    stat = stat.set("NDVI_quarter", ee.Algorithms.If(
        ee.Algorithms.IsEqual(ndvi_quarter, None),
        ee.Number(0),
        ndvi_quarter
    ))
    return ee.Feature(None, stat).set("millis", time_start)


def fc_to_dict(fc):
    """
    Earth EngineのFeatureCollectionを辞書型に変換する関数
    
    Args:
        fc: FeatureCollection
        
    Returns:
        辞書型のデータ（プロパティ名をキー、値のリストを値とする）
    """
    size = fc.size().getInfo()
    if size == 0:
        return {}
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()), selectors=prop_names
    ).get("list")
    return ee.Dictionary.fromLists(prop_names, prop_lists).getInfo()


def calc_ndvi(image):
    """
    NDVIを計算する関数
    NDVI = (NIR - RED) / (NIR + RED)
    
    Args:
        image: Sentinel-2画像（B4=Red, B8=NIR）
        
    Returns:
        NDVIバンドを含む画像
    """
    return ee.Image(
        image.expression(
            "(NIR-RED)/(NIR+RED)",
            {"RED": image.select("B4"), "NIR": image.select("B8")},
        )
    ).rename("NDVI").copyProperties(image, ["system:time_start"])


def ndvi_timeseries(data, title_suffix=""):
    """
    NDVI時系列グラフを生成する関数
    
    Args:
        data: pandas DataFrame（NDVIデータ）
        title_suffix: グラフタイトルのサフィックス
        
    Returns:
        Altairチャートオブジェクト
    """
    # データの存在確認
    has_mean = "NDVI_mean" in data.columns or "NDVI" in data.columns
    has_center = "NDVI_center" in data.columns
    has_quarter = "NDVI_quarter" in data.columns
    
    # 平均値の列名を決定
    mean_col = "NDVI_mean" if "NDVI_mean" in data.columns else "NDVI"
    
    # 日付列の準備
    if "Date" not in data.columns and "Timestamp" in data.columns:
        data["Date"] = pd.to_datetime(data["Timestamp"]).dt.date
    
    # 日付を文字列に変換（JSON serializationエラーを避けるため）
    if "Date" in data.columns:
        data["Date_str"] = data["Date"].astype(str)
    elif "Timestamp" in data.columns:
        data["Date_str"] = pd.to_datetime(data["Timestamp"]).dt.strftime("%Y-%m-%d")
    
    # データを縦長形式に変換（凡例表示のため）
    data_list = []
    if has_mean:
        mean_data = data[["Date_str", mean_col]].copy()
        mean_data = mean_data.rename(columns={mean_col: "NDVI_value"})
        mean_data["Type"] = "平均"
        data_list.append(mean_data)
    
    if has_center:
        center_data = data[["Date_str", "NDVI_center"]].copy()
        center_data = center_data.rename(columns={"NDVI_center": "NDVI_value"})
        center_data["Type"] = "中心点"
        data_list.append(center_data)
    
    if has_quarter:
        quarter_data = data[["Date_str", "NDVI_quarter"]].copy()
        quarter_data = quarter_data.rename(columns={"NDVI_quarter": "NDVI_value"})
        quarter_data["Type"] = "1/4正方形"
        data_list.append(quarter_data)
    
    # データを結合してグラフを作成
    if len(data_list) > 0:
        combined_data = pd.concat(data_list, ignore_index=True)
        
        # 色のマッピング
        color_scale = alt.Scale(
            domain=["平均", "中心点", "1/4正方形"],
            range=["blue", "red", "green"]
        )
        
        # グラフを作成
        chart = alt.Chart(combined_data).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("Date_str:N", title="観測日付", axis=alt.Axis(grid=True, labelAngle=-45)),
            y=alt.Y("NDVI_value:Q", title="NDVI", scale=alt.Scale(domain=[0.0, 1.0]), axis=alt.Axis(grid=True)),
            color=alt.Color("Type:N", scale=color_scale, legend=alt.Legend(title="種類", orient="top-left")),
            tooltip=[
                alt.Tooltip("Date_str:N", title="日付"),
                alt.Tooltip("NDVI_value:Q", title="NDVI", format=".4f"),
                alt.Tooltip("Type:N", title="種類"),
            ],
        ).properties(
            width=900,
            height=500,
            title={
                "text": f"NDVI時系列変化（2025年）{title_suffix}",
                "fontSize": 16,
                "fontWeight": "bold"
            }
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_legend(
            labelFontSize=12,
            titleFontSize=14
        ).interactive()
        
        return chart
    else:
        # フォールバック：平均値のみ
        if "Date_str" not in data.columns:
            if "Date" in data.columns:
                data["Date_str"] = data["Date"].astype(str)
            elif "Timestamp" in data.columns:
                data["Date_str"] = pd.to_datetime(data["Timestamp"]).dt.strftime("%Y-%m-%d")
        
        chart = alt.Chart(data).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("Date_str:N", title="観測日付", axis=alt.Axis(grid=True, labelAngle=-45)),
            y=alt.Y(f"{mean_col}:Q", title="NDVI", scale=alt.Scale(domain=[0.0, 1.0]), axis=alt.Axis(grid=True)),
            tooltip=[
                alt.Tooltip("Date_str:N", title="日付"),
                alt.Tooltip(f"{mean_col}:Q", title="NDVI", format=".4f"),
            ],
        ).properties(
            width=900,
            height=500,
            title={
                "text": f"NDVI時系列変化（2025年）{title_suffix}",
                "fontSize": 16,
                "fontWeight": "bold"
            }
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).interactive()
        return chart


# ============================================================================
# メイン処理：NDVI時系列データの取得と計算
# ============================================================================

# Sentinel-2画像コレクションの取得
Sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

# 各日付の画像を取得
collections = []
for date in target_dates:
    # 前後3日間の範囲で画像を取得
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")
    
    # 前後3日間の範囲で画像を取得
    date_collection_range = (
        Sentinel2
        .filterDate(f"{start_date}T00:00:00", f"{end_date}T23:59:59")
        .filterBounds(stat_region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))  # 雲量30%以下
        .map(maskS2clouds)
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

term_selected = ee.ImageCollection(collections[0])
for col in collections[1:]:
    term_selected = term_selected.merge(col)

# NDVIを計算
ndvi = term_selected.map(calc_ndvi)

# NDVI統計を取得
ndvi_stat_fc = ee.FeatureCollection(ndvi.map(reduce_region_function)).filter(
    ee.Filter.notNull(["NDVI"])
)

# 衛星データを辞書型に変換
ndvi_dict = fc_to_dict(ndvi_stat_fc)

if not ndvi_dict or len(ndvi_dict) == 0:
    raise ValueError("NDVIデータが取得できませんでした。")

# データフレームの作成
ndvi_df = pd.DataFrame(ndvi_dict)

# 配列の長さが異なる場合の処理（リストの場合は最初の要素を取得）
if len(ndvi_df) > 0:
    for col in ndvi_df.columns:
        if ndvi_df[col].dtype == 'object':
            try:
                ndvi_df[col] = ndvi_df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
            except:
                pass

# タイムスタンプを追加
ndvi_df["Timestamp"] = pd.to_datetime(ndvi_df["millis"], unit="ms")
ndvi_df["Date"] = ndvi_df["Timestamp"].dt.date
ndvi_df = ndvi_df.sort_values("Timestamp")

# 同じ日付の場合は最初の1枚のみを使用（重複除去）
ndvi_df_unique = ndvi_df.drop_duplicates(subset=["Date"], keep="first")
ndvi_df_unique = ndvi_df_unique.sort_values("Timestamp")

# 日付ごとのNDVI値を出力
for idx, row in ndvi_df_unique.iterrows():
    date_str = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "不明"
    ndvi_mean = row.get("NDVI_mean", row.get("NDVI", "N/A"))
    ndvi_center = row.get("NDVI_center", "N/A")
    ndvi_quarter = row.get("NDVI_quarter", "N/A")
    
    mean_str = f"{ndvi_mean:.4f}" if isinstance(ndvi_mean, (int, float)) else "N/A"
    center_str = f"{ndvi_center:.4f}" if isinstance(ndvi_center, (int, float)) else "N/A"
    quarter_str = f"{ndvi_quarter:.4f}" if isinstance(ndvi_quarter, (int, float)) else "N/A"
    print(f"{date_str},平均:{mean_str},中心:{center_str},1/4:{quarter_str}")

# 最初と最後の変化を計算
if len(ndvi_df_unique) >= 2:
    first_mean = ndvi_df_unique.iloc[0].get("NDVI_mean", ndvi_df_unique.iloc[0].get("NDVI"))
    last_mean = ndvi_df_unique.iloc[-1].get("NDVI_mean", ndvi_df_unique.iloc[-1].get("NDVI"))
    first_center = ndvi_df_unique.iloc[0].get("NDVI_center")
    last_center = ndvi_df_unique.iloc[-1].get("NDVI_center")
    first_quarter = ndvi_df_unique.iloc[0].get("NDVI_quarter")
    last_quarter = ndvi_df_unique.iloc[-1].get("NDVI_quarter")
    
    if isinstance(first_mean, (int, float)) and isinstance(last_mean, (int, float)):
        mean_diff = last_mean - first_mean
        print(f"平均変化:{mean_diff:.4f}")
    
    if isinstance(first_center, (int, float)) and isinstance(last_center, (int, float)):
        center_diff = last_center - first_center
        print(f"中心変化:{center_diff:.4f}")
    
    if isinstance(first_quarter, (int, float)) and isinstance(last_quarter, (int, float)):
        quarter_diff = last_quarter - first_quarter
        print(f"1/4変化:{quarter_diff:.4f}")

# CSVファイルに保存
ndvi_df.to_csv("ndvi_timeseries_original.csv", index=False)

# グラフ用に重複を除去したデータを使用
ndvi_df = ndvi_df_unique

# ============================================================================
# グラフの生成
# ============================================================================

if len(ndvi_df) > 0:
    # 全データのグラフ用データフレームの準備
    graph_df_all = ndvi_df.copy()
    if "Date" in graph_df_all.columns:
        graph_df_all = graph_df_all.drop(columns=["Date"])
    
    # 2025-04-21を除いたデータ
    exclude_date = pd.to_datetime("2025-04-21").date()
    graph_df_exclude_first = ndvi_df[ndvi_df["Date"] != exclude_date].copy()
    if "Date" in graph_df_exclude_first.columns:
        graph_df_exclude_first = graph_df_exclude_first.drop(columns=["Date"])
    
    # 全データのグラフを作成
    chart_all = ndvi_timeseries(graph_df_all, title_suffix="（全期間）")
    save(chart_all, "NDVI_timeseries_all.html")
    
    # 2025-04-21を除いたグラフを作成
    if len(graph_df_exclude_first) > 0:
        chart_exclude = ndvi_timeseries(graph_df_exclude_first, title_suffix="（4/21除く）")
        save(chart_exclude, "NDVI_timeseries_exclude_first.html")

# ============================================================================
# RGB画像マップの生成（2025-11-07）
# ============================================================================

try:
    # 2025-11-07の画像を取得（前後3日間の範囲で取得）
    target_date_rgb = "2025-11-07"
    date_obj_rgb = datetime.strptime(target_date_rgb, "%Y-%m-%d")
    start_date_rgb = (date_obj_rgb - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date_rgb = (date_obj_rgb + timedelta(days=3)).strftime("%Y-%m-%d")
    
    # 前後3日間の範囲で画像を取得
    date_collection_rgb = (
        Sentinel2
        .filterDate(f"{start_date_rgb}T00:00:00", f"{end_date_rgb}T23:59:59")
        .filterBounds(stat_region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(maskS2clouds)
        .sort("system:time_start")
    )
    
    count_rgb = date_collection_rgb.size().getInfo()
    if count_rgb > 0:
        # 対象日付に最も近い1枚を取得
        target_time_rgb = ee.Date(f"{target_date_rgb}T12:00:00").millis()
        date_collection_rgb = date_collection_rgb.map(
            lambda img: img.set("time_diff", ee.Number(img.get("system:time_start")).subtract(target_time_rgb).abs())
        ).sort("time_diff").limit(1)
    else:
        raise ValueError(f"2025-11-07の前後3日間（{start_date_rgb}～{end_date_rgb}）に画像が見つかりませんでした。")
    
    # RGB画像を指定範囲でクリップ
    rgb_image = date_collection_rgb.sort("system:time_start").first()
    rgb_bands = rgb_image.select(["B4", "B3", "B2"])  # Red, Green, Blue
    rgb_bands = rgb_bands.clip(stat_region)  # ポリゴン範囲でクリップ
    rgb_bands = ee.Image(rgb_bands).clip(stat_region)  # 確実にクリップ
    
    # ポリゴンの境界を計算（Folium用）
    polygon_coords_folium = [[coord[1], coord[0]] for coord in coords[:-1]]  # [緯度, 経度]に変換
    bounds = [[min(coord[1] for coord in coords[:-1]), min(coord[0] for coord in coords[:-1])],
              [max(coord[1] for coord in coords[:-1]), max(coord[0] for coord in coords[:-1])]]
    
    # Foliumマップを作成
    lat, lon = center_lat, center_lon
    rgb_map = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Map.add_ee_layer = add_ee_layer
    
    # RGB画像を追加
    visualization_RGB = {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]}
    rgb_map.add_ee_layer(rgb_bands, visualization_RGB, "RGB_2025/11/07")
    
    # ポリゴンの境界線と中心点を追加
    folium.Polygon(locations=polygon_coords_folium, color='red', weight=3, fillColor='red', fillOpacity=0.1).add_to(rgb_map)
    folium.Marker(location=[center_lat, center_lon], icon=folium.Icon(color='blue', icon='info-sign')).add_to(rgb_map)
    
    # 1/4サイズの正方形も追加
    quarter_coords_folium = [[coord[1], coord[0]] for coord in quarter_square_coords[:-1]]
    folium.Polygon(
        locations=quarter_coords_folium,
        color='green',
        weight=2,
        fillColor='green',
        fillOpacity=0.1,
        popup='1/4サイズの正方形',
        tooltip='1/4正方形範囲'
    ).add_to(rgb_map)
    
    # ポリゴン全体が表示されるようにfit_bounds
    rgb_map.fit_bounds(bounds, padding=(5, 5))
    
    # HTMLファイルを保存
    temp_html_rgb = "temp_rgb_map.html"
    rgb_map.save(temp_html_rgb)
    
    # Seleniumでスクリーンショットを取得
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=2000,2000')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f'file://{os.path.abspath(temp_html_rgb)}')
    
    # マップ要素が表示されるまで待機
    wait = WebDriverWait(driver, 30)
    map_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "folium-map")))
    
    # 画像読み込み待機
    time.sleep(20)
    
    # fit_boundsが正しく適用されるようにJavaScriptで再設定
    driver.execute_script("""
        var mapDiv = document.querySelector('.folium-map');
        if (mapDiv && mapDiv._leaflet_id) {
            var map = window[mapDiv._leaflet_id];
            if (map) {
                var bounds = [[%f, %f], [%f, %f]];
                map.fitBounds(bounds);
                map.invalidateSize();
            }
        }
    """ % (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
    time.sleep(5)
    
    # マップ要素のスクリーンショットを取得
    screenshot_rgb = map_element.screenshot_as_png
    driver.quit()
    
    # PNGとして保存
    Image.open(io.BytesIO(screenshot_rgb)).save("RGB_20251107_NDVI_points.png", "PNG", optimize=True, quality=95)
    
    # デバッグ用HTMLファイルも保存
    debug_html = "RGB_20251107_debug.html"
    if os.path.exists(temp_html_rgb):
        shutil.copy(temp_html_rgb, debug_html)
    
    if os.path.exists(temp_html_rgb):
        os.remove(temp_html_rgb)
    
    print(f"✓ 2025年11月7日のRGB画像を 'RGB_20251107_NDVI_points.png' に保存しました。")
    print(f"  ポリゴン範囲: 緯度 {bounds[0][0]:.6f}-{bounds[1][0]:.6f}, 経度 {bounds[0][1]:.6f}-{bounds[1][1]:.6f}")
    
except Exception as e:
    print(f"警告: RGB画像の生成に失敗しました: {str(e)}")

# ============================================================================
# 取得場所マップの生成
# ============================================================================

try:
    # 2025-11-07のRGB画像を取得（前後3日間の範囲で取得）
    target_date_location = "2025-11-07"
    date_obj_location = datetime.strptime(target_date_location, "%Y-%m-%d")
    start_date_location = (date_obj_location - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date_location = (date_obj_location + timedelta(days=3)).strftime("%Y-%m-%d")
    
    # 前後3日間の範囲で画像を取得
    date_collection_location = (
        Sentinel2
        .filterDate(f"{start_date_location}T00:00:00", f"{end_date_location}T23:59:59")
        .filterBounds(stat_region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(maskS2clouds)
        .sort("system:time_start")
    )
    
    count_location = date_collection_location.size().getInfo()
    if count_location > 0:
        # 対象日付に最も近い1枚を取得
        target_time_location = ee.Date(f"{target_date_location}T12:00:00").millis()
        date_collection_location = date_collection_location.map(
            lambda img: img.set("time_diff", ee.Number(img.get("system:time_start")).subtract(target_time_location).abs())
        ).sort("time_diff").limit(1)
    else:
        raise ValueError(f"2025-11-07の前後3日間（{start_date_location}～{end_date_location}）に画像が見つかりませんでした。")
    
    # RGB画像を取得
    rgb_image_location = date_collection_location.sort("system:time_start").first()
    rgb_bands_location = rgb_image_location.select(["B4", "B3", "B2"])
    
    # マップの中心点（ポリゴンの中心）
    map_center = [center_lat, center_lon]
    
    # Foliumマップを作成
    location_map = folium.Map(location=map_center, zoom_start=14)
    folium.Map.add_ee_layer = add_ee_layer
    
    # RGB画像を背景として追加
    visualization_RGB_location = {"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]}
    location_map.add_ee_layer(rgb_bands_location, visualization_RGB_location, "RGB背景")
    
    # ポリゴンの境界線を追加
    polygon_coords_folium = [[coord[1], coord[0]] for coord in coords[:-1]]
    bounds_location = [[min(coord[1] for coord in coords[:-1]), min(coord[0] for coord in coords[:-1])],
                       [max(coord[1] for coord in coords[:-1]), max(coord[0] for coord in coords[:-1])]]
    
    folium.Polygon(
        locations=polygon_coords_folium,
        color='red',
        weight=3,
        fillColor='red',
        fillOpacity=0.1,
        popup='取得範囲（ポリゴン）',
        tooltip='NDVI取得範囲'
    ).add_to(location_map)
    
    # 中心点を追加
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f'中心点<br>緯度: {center_lat:.6f}<br>経度: {center_lon:.6f}',
        tooltip='中心点',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(location_map)
    
    # 1/4サイズの正方形も表示
    quarter_coords_folium = [[coord[1], coord[0]] for coord in quarter_square_coords[:-1]]
    folium.Polygon(
        locations=quarter_coords_folium,
        color='green',
        weight=2,
        fillColor='green',
        fillOpacity=0.1,
        popup='1/4サイズの正方形',
        tooltip='1/4正方形範囲'
    ).add_to(location_map)
    
    # ポリゴン全体が表示されるようにfit_bounds
    location_map.fit_bounds(bounds_location, padding=(5, 5))
    
    # 一時的なHTMLファイルを作成
    temp_html = "temp_map.html"
    location_map.save(temp_html)
    
    # Seleniumでスクリーンショットを取得
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=2000,2000')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f'file://{os.path.abspath(temp_html)}')
    
    # マップ要素が表示されるまで待機
    wait = WebDriverWait(driver, 30)
    map_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "folium-map")))
    
    # 画像読み込み待機
    time.sleep(20)
    
    # fit_boundsが正しく適用されるようにJavaScriptで再設定
    driver.execute_script("""
        var mapDiv = document.querySelector('.folium-map');
        if (mapDiv && mapDiv._leaflet_id) {
            var map = window[mapDiv._leaflet_id];
            if (map) {
                var bounds = [[%f, %f], [%f, %f]];
                map.fitBounds(bounds);
                map.invalidateSize();
            }
        }
    """ % (bounds_location[0][0], bounds_location[0][1], bounds_location[1][0], bounds_location[1][1]))
    time.sleep(5)
    
    # マップ要素のスクリーンショットを取得
    screenshot = map_element.screenshot_as_png
    driver.quit()
    
    # 画像を保存
    img = Image.open(io.BytesIO(screenshot))
    output_path = "NDVI_location_map_RGB.png"
    img.save(output_path, "PNG", optimize=True, quality=95)
    
    # 一時ファイルを削除
    if os.path.exists(temp_html):
        os.remove(temp_html)
    
    print(f"✓ 取得場所のマップ（RGB背景）を '{output_path}' に保存しました。")
    print(f"  ポリゴン中心点: 緯度 {center_lat:.6f}, 経度 {center_lon:.6f}")
    print(f"  ポリゴン範囲: 経度 {140.048947:.6f} - {140.068946:.6f}, 緯度 {35.14381:.6f} - {35.161424:.6f}")
    
    # ============================================================================
    # NDVI画像を背景にした取得場所マップの生成
    # ============================================================================
    
    # NDVI画像を計算（元の画像から直接計算）
    # rgb_image_locationは元の画像なので、そこからNDVIを計算
    print("\n=== NDVI画像の計算デバッグ ===")
    print(f"元の画像のバンド: {rgb_image_location.bandNames().getInfo()}")
    
    ndvi_image_location = calc_ndvi(rgb_image_location)
    print(f"NDVI計算後のバンド: {ndvi_image_location.bandNames().getInfo()}")
    
    # NDVIバンドのみを選択してクリップ
    ndvi_image_location = ndvi_image_location.select("NDVI").clip(stat_region)
    
    # NDVI画像が正しく計算されているか確認
    print("NDVI画像の計算を確認中...")
    ndvi_bands = ndvi_image_location.bandNames().getInfo()
    print(f"NDVI画像のバンド: {ndvi_bands}")
    
    # NDVI画像の統計情報を取得（デバッグ用）
    try:
        ndvi_stats = ndvi_image_location.reduceRegion(
            reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), '', True),
            geometry=stat_region,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        print(f"NDVI統計情報: 最小={ndvi_stats.get('NDVI_min', 'N/A'):.4f}, 最大={ndvi_stats.get('NDVI_max', 'N/A'):.4f}, 平均={ndvi_stats.get('NDVI_mean', 'N/A'):.4f}")
    except Exception as e:
        print(f"警告: NDVI統計情報の取得に失敗: {str(e)}")
    
    print("=== デバッグ終了 ===\n")
    
    # Foliumマップを作成（ベースマップを非表示にする）
    location_map_ndvi = folium.Map(location=map_center, zoom_start=14, tiles=None)
    folium.Map.add_ee_layer = add_ee_layer
    
    # NDVI画像を背景として追加（確実にNDVI画像が表示されるように）
    visualization_NDVI_location = {
        "min": 0.0,
        "max": 1.0,
        "palette": ["white", "red", "yellow", "lightgreen", "green", "blue"]
    }
    # NDVI画像を確実に追加
    try:
        location_map_ndvi.add_ee_layer(ndvi_image_location, visualization_NDVI_location, "NDVI")
        print("✓ NDVIレイヤーを追加しました")
    except Exception as e:
        print(f"警告: NDVIレイヤーの追加に失敗しました: {str(e)}")
    
    # レイヤーコントロールを追加（NDVIレイヤーのみ表示）
    location_map_ndvi.add_child(folium.LayerControl(collapsed=False))
    
    # ポリゴンの境界線を追加
    folium.Polygon(
        locations=polygon_coords_folium,
        color='red',
        weight=3,
        fillColor='red',
        fillOpacity=0.1,
        popup='取得範囲（ポリゴン）',
        tooltip='NDVI取得範囲'
    ).add_to(location_map_ndvi)
    
    # 中心点を追加
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f'中心点<br>緯度: {center_lat:.6f}<br>経度: {center_lon:.6f}',
        tooltip='中心点',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(location_map_ndvi)
    
    # 1/4サイズの正方形も表示
    folium.Polygon(
        locations=quarter_coords_folium,
        color='green',
        weight=2,
        fillColor='green',
        fillOpacity=0.1,
        popup='1/4サイズの正方形',
        tooltip='1/4正方形範囲'
    ).add_to(location_map_ndvi)
    
    # ポリゴン全体が表示されるようにfit_bounds
    location_map_ndvi.fit_bounds(bounds_location, padding=(5, 5))
    
    # 一時的なHTMLファイルを作成
    temp_html_ndvi = "temp_map_ndvi.html"
    location_map_ndvi.save(temp_html_ndvi)
    
    # Seleniumでスクリーンショットを取得
    chrome_options_ndvi = Options()
    chrome_options_ndvi.add_argument('--headless')
    chrome_options_ndvi.add_argument('--no-sandbox')
    chrome_options_ndvi.add_argument('--disable-dev-shm-usage')
    chrome_options_ndvi.add_argument('--window-size=2000,2000')
    
    driver_ndvi = webdriver.Chrome(options=chrome_options_ndvi)
    driver_ndvi.get(f'file://{os.path.abspath(temp_html_ndvi)}')
    
    # マップ要素が表示されるまで待機
    wait_ndvi = WebDriverWait(driver_ndvi, 30)
    map_element_ndvi = wait_ndvi.until(EC.presence_of_element_located((By.CLASS_NAME, "folium-map")))
    
    # 画像読み込み待機
    time.sleep(20)
    
    # fit_boundsが正しく適用されるようにJavaScriptで再設定
    driver_ndvi.execute_script("""
        var mapDiv = document.querySelector('.folium-map');
        if (mapDiv && mapDiv._leaflet_id) {
            var map = window[mapDiv._leaflet_id];
            if (map) {
                var bounds = [[%f, %f], [%f, %f]];
                map.fitBounds(bounds);
                map.invalidateSize();
            }
        }
    """ % (bounds_location[0][0], bounds_location[0][1], bounds_location[1][0], bounds_location[1][1]))
    time.sleep(5)
    
    # マップ要素のスクリーンショットを取得
    screenshot_ndvi = map_element_ndvi.screenshot_as_png
    driver_ndvi.quit()
    
    # 画像を読み込み
    ndvi_img = Image.open(io.BytesIO(screenshot_ndvi))
    img_array_ndvi = np.array(ndvi_img)
    
    # 図を作成（画像と同じサイズ）
    fig_ndvi, ax_ndvi = plt.subplots(figsize=(img_array_ndvi.shape[1]/100, img_array_ndvi.shape[0]/100), dpi=100)
    ax_ndvi.imshow(img_array_ndvi)
    ax_ndvi.axis('off')
    
    # NDVIのカラーパレット
    colors_ndvi = ['white', 'red', 'yellow', 'lightgreen', 'green', 'blue']
    n_bins_ndvi = 256
    cmap_ndvi = mcolors.LinearSegmentedColormap.from_list('ndvi', colors_ndvi, N=n_bins_ndvi)
    
    # 画像の右側にカラーバーを追加
    norm_ndvi = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm_ndvi = plt.cm.ScalarMappable(cmap=cmap_ndvi, norm=norm_ndvi)
    sm_ndvi.set_array([])
    
    # カラーバーを画像上に配置（右側）
    # 画像の右側にスペースを確保してカラーバーを配置
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_ndvi)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar_ndvi = plt.colorbar(sm_ndvi, cax=cax, orientation='vertical')
    cbar_ndvi.set_label('NDVI', fontsize=14, fontweight='bold', rotation=270, labelpad=20)
    cbar_ndvi.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar_ndvi.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    cbar_ndvi.ax.tick_params(labelsize=12)
    
    # タイトルを追加
    ax_ndvi.text(0.5, 0.98, 'NDVI画像 (2025-11-07)',
            transform=ax_ndvi.transAxes,
            ha='center', va='top',
            fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2),
            family='sans-serif')
    
    # PNGとして保存
    plt.tight_layout(pad=0)
    output_path_ndvi = "NDVI_location_map.png"
    plt.savefig(output_path_ndvi, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    
    # 一時ファイルを削除
    if os.path.exists(temp_html_ndvi):
        os.remove(temp_html_ndvi)
    
    print(f"✓ 取得場所のマップ（NDVI背景、カラーバー付き）を '{output_path_ndvi}' に保存しました。")
    
except ImportError:
    # フォールバック: HTMLファイルとして保存
    map_center = [center_lat, center_lon]
    location_map = folium.Map(location=map_center, zoom_start=14)
    polygon_coords_folium = [[coord[1], coord[0]] for coord in coords[:-1]]
    folium.Polygon(
        locations=polygon_coords_folium,
        color='red',
        weight=3,
        fillColor='red',
        fillOpacity=0.1,
        popup='取得範囲（ポリゴン）',
        tooltip='NDVI取得範囲'
    ).add_to(location_map)
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f'中心点<br>緯度: {center_lat:.6f}<br>経度: {center_lon:.6f}',
        tooltip='中心点',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(location_map)
    location_map.save("NDVI_location_map.html")
    
except Exception as e:
    # フォールバック: HTMLファイルとして保存
    map_center = [center_lat, center_lon]
    location_map = folium.Map(location=map_center, zoom_start=14)
    polygon_coords_folium = [[coord[1], coord[0]] for coord in coords[:-1]]
    folium.Polygon(
        locations=polygon_coords_folium,
        color='red',
        weight=3,
        fillColor='red',
        fillOpacity=0.1,
        popup='取得範囲（ポリゴン）',
        tooltip='NDVI取得範囲'
    ).add_to(location_map)
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f'中心点<br>緯度: {center_lat:.6f}<br>経度: {center_lon:.6f}',
        tooltip='中心点',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(location_map)
    location_map.save("NDVI_location_map.html")
