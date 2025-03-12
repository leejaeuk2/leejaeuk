import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# 파일 선택 함수
def select_file():
    root = Tk()
    root.withdraw()  # Tk 창 숨기기
    return filedialog.askopenfilename(title="CSV 파일을 선택하세요", filetypes=[("CSV files", "*.csv")])

# 사용자에게 파일 선택 요청
print("첫 번째 CSV 파일을 선택하세요.")
file1_path = select_file()
print("두 번째 CSV 파일을 선택하세요.")
file2_path = select_file()

# 파일 로드
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# NaN 값 보간 (선형 보간)
df2_interpolated = df2.interpolate(method="linear")

# 차이 계산 (df1 - df2)
diff = df1.set_index("frame").subtract(df2.set_index("frame"))

# 프레임 번호 (x축)
frames = diff.index

# 각 조인트별 유클리드 거리 차이 계산
joints = ["IL", "HP", "KN", "AK", "TO"]
diff_values = {joint: np.sqrt(diff[f"{joint}_x"]**2 + diff[f"{joint}_y"]**2) for joint in joints}

# 각 조인트별 평균 차이 계산
mean_values = {joint: np.mean(diff_values[joint]) for joint in joints}

# 전체 평균 차이 계산
mean_diff = np.mean(list(diff_values.values()), axis=0)
overall_mean = np.mean(mean_diff)

# y축 범위 설정 (최대값의 1.2배로 제한)
y_max = 60

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 각 조인트별 차이 플롯
for joint in joints:
    plt.plot(frames, diff_values[joint], label=f"{joint} (avg: {mean_values[joint]:.2f})")

# 평균 차이 그래프 추가 (점선)
plt.plot(frames, mean_diff, linestyle="dashed", color="black", label=f"Mean Difference (avg: {overall_mean:.2f})")

plt.xlabel("Frame")
plt.ylabel("Coordinate Difference (Euclidean Distance)")
plt.title("Difference in Joint Coordinates Between Two Models")
plt.ylim(0, y_max)  # y축 범위 설정
plt.legend()
plt.grid(True)
plt.show()
