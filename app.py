from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np  # Đảm bảo có numpy
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Đảm bảo import này

# --- 1. Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="API Dự Đoán Khách Hàng Vỡ Nợ",
    description="API này dự đoán khả năng một khách hàng vay tín dụng sẽ vỡ nợ dựa trên thông tin cá nhân và lịch sử tín dụng của họ.",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

# ... (các phần khác của FastAPI app) ...

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        "file://", # Giữ lại nếu bạn có thể mở bằng cách kéo thả file
        "null",    # Giữ lại cho trường hợp file://
        "http://localhost:63342" # ĐẢM BẢO DÒNG NÀY CÓ VÀ ĐÚNG CHÍNH TẢ
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (các phần còn lại của app.py) ... --- Kết thúc cấu hình CORS ---

# --- 2. Tải lại mô hình và trình tiền xử lý (ColumnTransformer) ---
MODEL_PATH = 'xgboost_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'

model = None
preprocessor = None

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Mô hình và preprocessor đã được tải thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại '{MODEL_PATH}' hoặc '{PREPROCESSOR_PATH}'.")
    print("Đảm bảo bạn đã tải chúng từ Colab về và đặt đúng thư mục trong PyCharm.")
    raise RuntimeError("Không thể tải các file cần thiết cho API. Vui lòng kiểm tra lại đường dẫn file.")
except Exception as e:
    print(f"Lỗi khi tải mô hình/preprocessor: {e}")
    raise RuntimeError("Lỗi không xác định khi tải tài nguyên.")


# --- 3. Định nghĩa cấu trúc dữ liệu đầu vào (Input Schema - Dữ liệu GỐC) ---
# Lớp này phải chứa TẤT CẢ các cột dữ liệu GỐC mà bạn cần để tạo ra các features phái sinh.
class CustomerRawFeatures(BaseModel):
    LIMIT_BAL: float = Field(..., example=200000, description="Hạn mức tín dụng được cấp (TWD)")
    SEX: int = Field(..., example=2, description="Giới tính (1 = Nam, 2 = Nữ)")
    EDUCATION: int = Field(..., example=2, description="Trình độ học vấn (1=Sau ĐH, 2=ĐH, 3=PTTH, 4=Khác)")
    MARRIAGE: int = Field(..., example=1, description="Tình trạng hôn nhân (1=Độc thân, 2=Đã kết hôn, 3=Khác)")
    AGE: int = Field(..., example=35, description="Tuổi của khách hàng")

    # PAY_x: Tình trạng thanh toán 6 tháng gần nhất (PAY_1: tháng 9, ..., PAY_6: tháng 4)
    PAY_1: int = Field(..., example=0, description="Tình trạng thanh toán tháng 9")
    PAY_2: int = Field(..., example=0, description="Tình trạng thanh toán tháng 8")
    PAY_3: int = Field(..., example=0, description="Tình trạng thanh toán tháng 7")
    PAY_4: int = Field(..., example=0, description="Tình trạng thanh toán tháng 6")
    PAY_5: int = Field(..., example=0, description="Tình trạng thanh toán tháng 5")
    PAY_6: int = Field(..., example=0, description="Tình trạng thanh toán tháng 4")

    # BILL_AMT_x: Số tiền hóa đơn chưa thanh toán 6 tháng gần nhất
    BILL_AMT1: float = Field(..., example=3913, description="Số tiền hóa đơn tháng 9")
    BILL_AMT2: float = Field(..., example=3102, description="Số tiền hóa đơn tháng 8")
    BILL_AMT3: float = Field(..., example=689, description="Số tiền hóa đơn tháng 7")
    BILL_AMT4: float = Field(..., example=0, description="Số tiền hóa đơn tháng 6")
    BILL_AMT5: float = Field(..., example=0, description="Số tiền hóa đơn tháng 5")
    BILL_AMT6: float = Field(..., example=0, description="Số tiền hóa đơn tháng 4")

    # PAY_AMT_x: Số tiền khách hàng đã thanh toán 6 tháng gần nhất
    PAY_AMT1: float = Field(..., example=0, description="Số tiền đã trả tháng 9")
    PAY_AMT2: float = Field(..., example=0, description="Số tiền đã trả tháng 8")
    PAY_AMT3: float = Field(..., example=0, description="Số tiền đã trả tháng 7")
    PAY_AMT4: float = Field(..., example=0, description="Số tiền đã trả tháng 6")
    PAY_AMT5: float = Field(..., example=0, description="Số tiền đã trả tháng 5")
    PAY_AMT6: float = Field(..., example=0, description="Số tiền đã trả tháng 4")


# --- 4. Định nghĩa Endpoint API ---
@app.post("/predict_default")
def predict_default(raw_features: CustomerRawFeatures):
    """
    Dự đoán khả năng vỡ nợ của một khách hàng dựa trên dữ liệu gốc được cung cấp.
    API này sẽ tự động tái tạo các đặc trưng phái sinh và tiền xử lý dữ liệu trước khi dự đoán.
    Trả về dự đoán (0: không vỡ nợ, 1: vỡ nợ) và xác suất vỡ nợ của từng lớp.
    """
    try:
        # Chuyển dữ liệu đầu vào thô từ Pydantic model sang Pandas DataFrame
        # Đảm bảo thứ tự cột ở đây khớp với thứ tự trong dữ liệu gốc của bạn nếu bạn dùng df[original_columns]
        # (Ở đây ta dùng df[col] = raw_features.col để đảm bảo thứ tự)
        data = raw_features.dict()
        df_raw = pd.DataFrame([data])

        # ============ TÁI TẠO CÁC FEATURE ENGINEERING TẠI ĐÂY ============
        # PHẢI LÀM Y HỆT CÁC BƯỚC BẠN ĐÃ LÀM TRONG COLAB ĐỂ TẠO CÁC FEATURES PHÁI SINH
        # Dựa trên code của bạn, đây là các features bạn đã đưa vào X trong Colab:

        # 1. no_payment_flag (nếu PAY_x == -2)
        df_raw['no_payment_flag'] = (
                (df_raw['PAY_1'] == -2) | (df_raw['PAY_2'] == -2) |
                (df_raw['PAY_3'] == -2) | (df_raw['PAY_4'] == -2) |
                (df_raw['PAY_5'] == -2) | (df_raw['PAY_6'] == -2)
        ).astype(int)

        # 2. late_but_no_balance_flag (nếu PAY_x > 0 và BILL_AMT_x == 0)
        df_raw['late_but_no_balance_flag'] = (
                ((df_raw['PAY_1'] > 0) & (df_raw['BILL_AMT1'] == 0)) |
                ((df_raw['PAY_2'] > 0) & (df_raw['BILL_AMT2'] == 0)) |
                ((df_raw['PAY_3'] > 0) & (df_raw['BILL_AMT3'] == 0)) |
                ((df_raw['PAY_4'] > 0) & (df_raw['BILL_AMT4'] == 0)) |
                ((df_raw['PAY_5'] > 0) & (df_raw['BILL_AMT5'] == 0)) |
                ((df_raw['PAY_6'] > 0) & (df_raw['BILL_AMT6'] == 0))
        ).astype(int)

        # 3. COUNT_LATE (Đếm số lần trễ hạn > 0)
        pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        df_raw['COUNT_LATE'] = df_raw[pay_cols].apply(lambda row: (row > 0).sum(), axis=1)

        # 4. MAX_DELAY (Độ trễ tối đa)
        df_raw['MAX_DELAY'] = df_raw[pay_cols].max(axis=1)

        # 5. TOTAL_PAYMENT (Tổng số tiền đã trả)
        pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        df_raw['TOTAL_PAYMENT'] = df_raw[pay_amt_cols].sum(axis=1)

        # 6. LIMIT_BAL_LOG (Log của hạn mức tín dụng)
        df_raw['LIMIT_BAL_LOG'] = np.log1p(df_raw['LIMIT_BAL'])

        # 7. RECENCY_WEIGHTED_DELAY (Tính toán theo trọng số thời gian)
        # Giả sử bạn có trọng số: W = [6, 5, 4, 3, 2, 1] cho PAY_1 đến PAY_6
        weights = np.array([6, 5, 4, 3, 2, 1])
        df_raw['RECENCY_WEIGHTED_DELAY'] = (df_raw[pay_cols] * weights).sum(axis=1)

        # 8. overpay_flag (kiểm tra nếu PAY_AMT_x > BILL_AMT_x)
        df_raw['overpay_flag'] = (
                ((df_raw['PAY_AMT1'] > df_raw['BILL_AMT1']) & (df_raw['BILL_AMT1'] > 0)) |
                ((df_raw['PAY_AMT2'] > df_raw['BILL_AMT2']) & (df_raw['BILL_AMT2'] > 0)) |
                ((df_raw['PAY_AMT3'] > df_raw['BILL_AMT3']) & (df_raw['BILL_AMT3'] > 0)) |
                ((df_raw['PAY_AMT4'] > df_raw['BILL_AMT4']) & (df_raw['BILL_AMT4'] > 0)) |
                ((df_raw['PAY_AMT5'] > df_raw['BILL_AMT5']) & (df_raw['BILL_AMT5'] > 0)) |
                ((df_raw['PAY_AMT6'] > df_raw['BILL_AMT6']) & (df_raw['BILL_AMT6'] > 0))
        ).astype(int)

        # 9. DEBT_TO_LIMIT_RATIO (Tỷ lệ nợ trên hạn mức)
        bill_amt_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        df_raw['TOTAL_BILL_AMT'] = df_raw[bill_amt_cols].sum(axis=1)
        df_raw['DEBT_TO_LIMIT_RATIO'] = df_raw.apply(
            lambda row: row['TOTAL_BILL_AMT'] / row['LIMIT_BAL'] if row['LIMIT_BAL'] != 0 else 0, axis=1
        )
        # Xóa cột tạm TOTAL_BILL_AMT nếu không cần
        df_raw = df_raw.drop(columns=['TOTAL_BILL_AMT'])

        # ============ KẾT THÚC FEATURE ENGINEERING ============

        # Đây là danh sách 'features' mà bạn đã dùng để tạo X = df[features] trong Colab
        # Đảm bảo thứ tự này khớp với thứ tự các cột mà ColumnTransformer đã được FIT.
        # Danh sách này phải giống hệt danh sách `features` bạn đã định nghĩa trong Colab.
        final_features_for_preprocessor = [
            'no_payment_flag',
            'late_but_no_balance_flag',
            'EDUCATION',  # Categorical
            'COUNT_LATE',
            'MAX_DELAY',
            'PAY_1',  # Categorical
            'PAY_2',  # Categorical
            'TOTAL_PAYMENT',
            'LIMIT_BAL_LOG',
            'RECENCY_WEIGHTED_DELAY',
            'SEX',  # Categorical
            'AGE',
            'MARRIAGE',  # Categorical
            'overpay_flag',
            'DEBT_TO_LIMIT_RATIO',
            'BILL_AMT5'
        ]

        # Trích xuất các cột cần thiết cho preprocessor theo đúng thứ tự
        X_for_preprocessor = df_raw[final_features_for_preprocessor]

        # Áp dụng ColumnTransformer đã được huấn luyện
        # preprocessor.transform sẽ tự động xử lý StandardScaler và OneHotEncoder
        input_processed_array = preprocessor.transform(X_for_preprocessor)

        # Dự đoán
        prediction = model.predict(input_processed_array)[0]
        probability = model.predict_proba(input_processed_array)[0].tolist()

        return {
            "prediction": int(prediction),
            "probability_no_default": probability[0],
            "probability_default": probability[1],
            "message": "Khách hàng có khả năng vỡ nợ" if prediction == 1 else "Khách hàng ít khả năng vỡ nợ"
        }
    except Exception as e:
        print(f"Đã xảy ra lỗi trong hàm predict_default: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình dự đoán: {str(e)}")


# Endpoint đơn giản để kiểm tra API có đang chạy hay không
@app.get("/")
def read_root():
    return {"message": "API Dự Đoán Khách Hàng Vỡ Nợ đang chạy và sẵn sàng!"}