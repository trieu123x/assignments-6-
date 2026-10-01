# Intelligent System Lab — Assignment 01 → 06

**Đinh Hải Triều — B23DCCN843 — Lớp 06** · Intelligent System Development · TS. Trần Đình Quế

Một site tĩnh gộp mọi web demo **serverless** của môn học. Mỗi bài nằm trong một thư mục
con độc lập. Không có máy chủ suy luận: trọng số mô hình là file JSON/ONNX tĩnh, trình duyệt
tự tính toán.

| Đường dẫn | Bài | Nội dung | Suy luận trong trình duyệt |
|---|---|---|---|
| `/` | — | trang hub | — |
| `/a06-rnn/` | A06 | RNN, LSTM, GRU: Churn Modelling, giá cổ phiếu Amazon | `rnn.js` (JavaScript thuần, viết từ công thức) |
| `/a05-cnn/` | A05 | LeNet-5, AlexNet, VGG-16, ResNet-50: CIFAR-10, Flowers; CNN-1D: Diabetes | ONNX Runtime Web (WASM), 10 mô hình int8 |
| `/a02-recommend/` | A02 (+A03, A04) | review thương mại điện tử: LogReg TF-IDF, MLP, Text-CNN | JavaScript (React/Vite) |
| `/a01-tieu-duong/` | A01 (+A03, A04) | Pima Diabetes: Random Forest, MLP, CNN-1D | JavaScript |
| `/a01-gia-nha/` | A01 (+A03, A04) | giá nhà Việt Nam 2024: Gradient Boosting, MLP, CNN | JavaScript |

## Deploy lên Vercel

Không cần build. Vào vercel.com/new, chọn **Import** repo này, đặt Framework Preset là
**Other**, để trống Build Command, đặt Output Directory là `.`, rồi bấm **Deploy**.

Hoặc dùng CLI: `npx vercel deploy --prod`.

`vercel.json` tắt `cleanUrls`: các site con dùng liên kết tương đối có tên file
(`cifar10.html`), nên chúng chạy đúng khi nằm trong thư mục con. `/a05-cnn` được chuyển
hướng sang `/a05-cnn/` để đường dẫn tương đối không bị lệch.

## Dựng lại

Site được sinh tự động từ mã nguồn của từng tuần bằng `tuan 6/lib/build_hub.py`. Script
sao chép, build lại ứng dụng Vite với `--base ./`, sửa liên kết, và chèn nút "← Tất cả bài tập"
vào mỗi trang:

```bash
py -3.11 "tuan 6/lib/build_hub.py" assignments-hub
node "tuan 6/lib/check_hub.mjs" assignments-hub     # kiểm thử mọi trang trong Edge headless
```
