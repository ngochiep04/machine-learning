import streamlit as st
import pandas as pd
import joblib

# Tải mô hình và các thành phần liên quan

from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model("xgb_model.json")
model.n_classes_ = 3
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Dự đoán khách hàng", layout="wide")

# Nhãn hiển thị cho các trường
label_map = {
    "Tuoi": "Tuổi",
    "Thu_nhap": "Thu nhập",
    "Chi_tieu": "Chi tiêu",
    "Tien_dien": "Tiền điện",
    "So_tien_vay": "Số tiền vay",
    "Th_kvay": "Thời hạn khoản vay",
    "Nguoi_phu_thuoc": "Số người phụ thuộc",
    "Tt_cv": "Tình trạng công việc",
    "Tt_hon_nhan": "Tình trạng hôn nhân",
    "Ts_the_chap": "Tài sản thế chấp",
    "Tt_sh_nha": "Tình trạng sở hữu nhà",
    "Md_vay": "Mục đích vay",
    "Cm_thu_nhap": "Chứng minh thu nhập",
    "Td_hoc_van": "Trình độ học vấn"
}

# Các đặc trưng nhập số
numeric_features = [
    "Tuoi", "Thu_nhap", "Chi_tieu", "Tien_dien",
    "So_tien_vay", "Th_kvay", "Nguoi_phu_thuoc"
]

# Các đặc trưng phân loại cần ánh xạ
categorical_mappings = {
    "Tt_cv": [str(i) for i in range(1, 21)],
    "Tt_hon_nhan": {
        "Có gia đình": 1,
        "Độc thân": 2,
        "Ly hôn": 3,
        "Góa": 4
    },
    "Ts_the_chap": {
        "Có tài sản thế chấp": 0,
        "Không có tài sản thế chấp": 1
    },
    "Tt_sh_nha": {
        "Đã sở hữu nhà ở": 0,
        "Chưa sở hữu nhà ở": 1
    },
    "Md_vay": {
        "Tiêu dùng": 1,
        "Học tập": 2,
        "Sản xuất kinh doanh": 3,
        "Mua xe": 4,
        "Mua nhà": 5,
        "Đầu tư chứng khoán": 6
    },
    "Cm_thu_nhap": {
        "Có giấy tờ chứng minh": 0,
        "Không có giấy tờ chứng minh": 1
    },
    "Td_hoc_van": {
        "Tiến sĩ": 1,
        "Thạc sĩ": 2,
        "Đại học": 3,
        "Cấp ba": 4
    }
}

# Giao diện nhập liệu
st.subheader("📥 Nhập thông tin khách hàng")
user_input = {}

# Nhập số: chia thành 2 dòng
cols_num = st.columns(4)
for i, feature in enumerate(numeric_features):
    with cols_num[i % 4]:
        user_input[feature] = st.number_input(label_map[feature], value=0, step=1)

# Chọn dropdown: chia thành 3–4 dòng
cat_features = list(categorical_mappings.keys())
for row in range(0, len(cat_features), 3):
    cols_cat = st.columns(3)
    for i, feature in enumerate(cat_features[row:row+3]):
        with cols_cat[i]:
            options = list(categorical_mappings[feature].keys()) if isinstance(categorical_mappings[feature], dict) else categorical_mappings[feature]
            selected = st.selectbox(label_map[feature], ["Chọn"] + options)
            if selected != "Chọn":
                user_input[feature] = categorical_mappings[feature][selected] if isinstance(categorical_mappings[feature], dict) else int(selected)
            else:
                user_input[feature] = ""

# Nút dự đoán: đặt giữa và làm to
st.markdown("<br>", unsafe_allow_html=True)
centered_button = st.columns([1, 2, 1])[1]
with centered_button:
    if st.button("🔍 DỰ ĐOÁN", use_container_width=True):
        missing_count = sum(v == "" or v == 0 for v in user_input.values())

        if missing_count == len(user_input):
            st.error("⚠️ Hãy nhập thông tin khách hàng.")
        else:
            if missing_count > 3:
                st.warning("⚠️ Bạn đang bỏ trống quá nhiều thông tin, kết quả dự đoán có thể không chính xác.")

            # Xử lý dữ liệu đầu vào
            input_df = pd.DataFrame([user_input])
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
            input_encoded = input_encoded.fillna(0)
            input_encoded = input_encoded.astype(float)
            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)[0]

            label_mapping = {
                0: "Không thể trả nợ",
                1: "Có thể trả nợ",
                2: "Chắc chắn trả được nợ"
            }

            st.success(f"✅ Kết quả dự đoán: {label_mapping[prediction]}")