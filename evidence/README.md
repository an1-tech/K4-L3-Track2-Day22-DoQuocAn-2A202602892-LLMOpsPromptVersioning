# Phân tích V1 và V2 — Day22

**Học viên:** Đỗ Quốc An — **MSSV:** 2A202602892  
**LangSmith project:** `day22-lab`

## Thiết lập so sánh

- **V1** (`doquocan-2a202602892-day22-rag-v1`): trả lời trực tiếp, ngắn gọn trong 2–4 câu, chỉ sử dụng thông tin được context hỗ trợ.
- **V2** (`doquocan-2a202602892-day22-rag-v2`): trả lời có cấu trúc trong 3–5 câu, trình bày ý chính, cơ chế và mục đích hoặc giới hạn khi context có thông tin.
- Cả hai được đánh giá trên **cùng 50 cặp QA**, tổng cộng **100 câu trả lời**. Reference chỉ dùng để đánh giá, không đưa vào prompt sinh câu trả lời.
- Cấu hình: OpenAI `gpt-4o-mini`, embedding `text-embedding-3-small`, FAISS, chunk size **500 ký tự**, overlap **50 ký tự**, retrieve **top-3**. Model và cấu hình retrieval được giữ giống nhau giữa hai phiên bản.

## Kết quả RAGAS

Số liệu lấy từ [báo cáo JSON](03_ragas_report.json), làm tròn đến bốn chữ số thập phân.

| Metric | V1 | V2 | Kết quả trong lần đánh giá này |
|---|---:|---:|---|
| Faithfulness | **0.9680** | 0.7670 | V1 cao hơn |
| Answer relevancy | **0.9175** | 0.8767 | V1 cao hơn |
| Context recall | 1.0000 | 1.0000 | Bằng nhau |
| Context precision | 0.9417 | **0.9450** | V2 nhỉnh hơn |

**Đạt yêu cầu bài:** V1 có faithfulness **0.9680 ≥ 0.8**. V2 chưa đạt ngưỡng này; yêu cầu bắt buộc chỉ cần ít nhất một phiên bản đạt.

## Phân tích

V1 có faithfulness cao hơn V2 khoảng **0.2010** và answer relevancy cao hơn khoảng **0.0408**. Kết quả này phù hợp với giả thuyết rằng câu trả lời ngắn giúp giảm các diễn giải ngoài context và tập trung hơn vào câu hỏi.

Ví dụ thực tế với câu hỏi **“What is Constitutional AI?”**: V1 mô tả các nguyên tắc, bước phê bình và chỉnh sửa, cùng việc giảm nhu cầu phản hồi của con người; các ý này có trong context được retrieve. V2 bổ sung nhận định rằng hiệu quả có thể phụ thuộc vào độ phức tạp của nguyên tắc và khả năng tự đánh giá của model. Nhận định bổ sung này không xuất hiện trong ba đoạn context của mẫu đó, minh họa nguy cơ phát sinh phát biểu thiếu căn cứ khi yêu cầu câu trả lời chi tiết hơn. Có thể đối chiếu trong [đầu ra V1](../data/rag_outputs_v1.json) và [đầu ra V2](../data/rag_outputs_v2.json).

Context recall bằng **1.0** ở cả hai phiên bản. Context precision chỉ chênh khoảng **0.0033**. Vì cấu hình retrieval giống nhau và các metric này chủ yếu đánh giá tài liệu truy xuất, không thể kết luận prompt V2 cải thiện retrieval chỉ từ chênh lệch nhỏ này; biến thiên của evaluator có thể đóng góp vào kết quả.

Trong phạm vi lần thử nghiệm này, **ưu tiên V1** khi mục tiêu là trả lời ngắn, bám sát nguồn và đúng trọng tâm. Nếu tiếp tục cải thiện V2, cần siết yêu cầu mọi nhận định về cơ chế, mục đích hoặc giới hạn đều có căn cứ trong context, rồi đánh giá lại trên cùng 50 QA.

Đây là một lần đánh giá bằng LLM, chưa đủ để kết luận khác biệt có ý nghĩa thống kê. Faithfulness phản ánh mức độ bám context, không tự chứng minh context đúng hoặc cập nhật với thực tế.

## Bằng chứng

- [RAG + LangSmith traces](01_langsmith_traces.png).
- [Hai prompt trên Hub](02_prompt_hub.png) và [log A/B routing](02_ab_routing_log.txt): V1 nhận **19** câu, V2 nhận **31** câu trong bước routing. Đây là tập chạy riêng; bước RAGAS vẫn đánh giá đủ **50 câu mỗi phiên bản**.
- [Bảng điểm RAGAS](03_ragas_scores.png) và [log evaluation](03_ragas_run_log.txt).
- [PII demo](04_pii_demo_log.txt): **6/6** case kiểm tra output đạt.
- [JSON demo](04_json_demo_log.txt): **5/5** case kiểm tra output đạt, bao gồm error fallback cho đầu vào không thể sửa.

Log Guardrails có cảnh báo gửi telemetry do lỗi phân giải tên miền. Các kiểm tra output vẫn đạt; cảnh báo này được tài liệu `CHECKPOINTS.md` cho phép bỏ qua.
