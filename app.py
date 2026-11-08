import os
import zipfile
import joblib
import gdown
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

# =========================
# 1. TẢI MÔ HÌNH TỪ GOOGLE DRIVE
# =========================

MODEL_ID = "1guga23421pH92MPfKpba2hohCHlp4Xmk"

if not os.path.exists("SMART_ENERGY_MODELS.zip"):
    gdown.download(
        id=MODEL_ID,
        output="SMART_ENERGY_MODELS.zip",
        quiet=False
    )

with zipfile.ZipFile("SMART_ENERGY_MODELS.zip", "r") as z:
    z.extractall("models")

rf_model = joblib.load("models/smart_energy_rf.joblib")
ann_model = joblib.load("models/smart_energy_ann.joblib")

# =========================
# 2. TẢI DỮ LIỆU EXCEL
# =========================

DATA_ID = "1OWV3RzEsFAfoalWGsgVKovIjkd9L3B4x"

DATA_URL = (
    f"https://docs.google.com/spreadsheets/d/{DATA_ID}/export?format=xlsx"
)

df = pd.read_excel(DATA_URL)

FEATURES = [
    "hour",
    "day",
    "month",
    "dayofweek",
    "total load forecast",
    "forecast solar day ahead",
    "forecast wind onshore day ahead"
]

# =========================
# 3. HÀM DỰ ĐOÁN
# =========================

def predict_energy(model_name, date, time):

    try:
        selected_date = pd.to_datetime(date)
        selected_time = pd.to_datetime(time).time()

        hour = selected_time.hour

        day = selected_date.day
        month = selected_date.month
        dayofweek = selected_date.dayofweek

        matched = df[
            (df["hour"] == hour) &
            (df["day"] == day) &
            (df["month"] == month) &
            (df["dayofweek"] == dayofweek)
        ]

        if matched.empty:
            return "Không tìm thấy dữ liệu phù hợp.", None

        # Lấy giá trị trung bình của các thông số dự báo
        input_data = matched[FEATURES].mean().to_frame().T

        # Chọn mô hình
        if model_name == "Random Forest":
            model = rf_model
        else:
            model = ann_model

        prediction = float(model.predict(input_data)[0])

        # Biểu đồ
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["Nhu cầu tiêu thụ điện dự đoán"],
            y=[prediction],
            text=[f"{prediction:,.2f} MW"],
            textposition="auto"
        ))

        fig.update_layout(
            title="KẾT QUẢ DỰ ĐOÁN SMART ENERGY",
            yaxis_title="Công suất (MW)",
            template="plotly_white"
        )

        result = (
            f"Ngày: {selected_date.strftime('%d/%m/%Y')}\n"
            f"Giờ: {selected_time.strftime('%H:%M')}\n"
            f"Mô hình: {model_name}\n"
            f"Nhu cầu tiêu thụ điện dự đoán: {prediction:,.2f} MW"
        )

        return result, fig

    except Exception as e:
        return f"Có lỗi xảy ra: {str(e)}", None

# =========================
# 4. GIAO DIỆN ỨNG DỤNG
# =========================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="SMART ENERGY"
) as app:

    gr.Markdown("""
    # ⚡ SMART ENERGY
    ### HỆ THỐNG DỰ ĐOÁN NHU CẦU TIÊU THỤ ĐIỆN

    Ứng dụng sử dụng thuật toán Random Forest và ANN.
    """)

    with gr.Row():
        model_input = gr.Dropdown(
            ["Random Forest", "ANN"],
            value="Random Forest",
            label="Chọn thuật toán"
        )

    with gr.Row():
        date_input = gr.Textbox(
            label="Ngày (YYYY-MM-DD)",
            placeholder="Ví dụ: 2017-01-15"
        )

        time_input = gr.Textbox(
            label="Giờ (HH:MM)",
            placeholder="Ví dụ: 08:00"
        )

    predict_button = gr.Button(
        "⚡ DỰ ĐOÁN",
        variant="primary"
    )

    result_output = gr.Textbox(
        label="Kết quả dự đoán",
        lines=4
    )

    chart_output = gr.Plot(
        label="Biểu đồ kết quả"
    )

    predict_button.click(
        fn=predict_energy,
        inputs=[
            model_input,
            date_input,
            time_input
        ],
        outputs=[
            result_output,
            chart_output
        ]
    )

    gr.Markdown("""
    ---
    **SMART ENERGY – NHÓM 2**

    Dự án ứng dụng học máy trong dự đoán nhu cầu tiêu thụ điện.
    """)

app.launch(server_name="0.0.0.0", server_port=7860)
