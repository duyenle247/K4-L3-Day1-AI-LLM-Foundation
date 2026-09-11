# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

- Khi `temperature = 0.0`: Phản hồi mang tính tất định (deterministic), tập trung vào sự kiện phổ biến nhất (như xuất khẩu cà phê hoặc hang Sơn Đoòng), câu văn chuẩn tắc và nội dung lặp lại y hệt qua các lần gọi.
- Khi `temperature = 0.5 – 1.0`: Câu trả lời có sự đa dạng hơn về góc nhìn và vốn từ vựng, diễn đạt tự nhiên hơn trong khi vẫn giữ vững tính logic và độ chính xác thực tế.
- Khi `temperature = 1.5`: Phân phối xác suất token bị làm phẳng khiến mô hình chọn các từ ngữ bất thường, câu văn trở nên bay bổng quá mức, thiếu mạch lạc hoặc dễ xuất hiện ảo giác (hallucination).
=> Quy luật: Temperature tỷ lệ thuận với độ ngẫu nhiên; giá trị càng thấp câu trả lời càng ổn định và chính xác, giá trị càng cao phản hồi càng sáng tạo nhưng độ tin cậy và mạch lạc giảm dần.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt `temperature` ở mức thấp, trong khoảng **0.0 đến 0.2** (thậm chí là **0.0** nếu kết hợp RAG/tra cứu FAQ).
> Lý do:
> 1. Triệt tiêu tối đa ảo giác (hallucination), đảm bảo thông tin chính sách, bảo hành, bảng giá bám sát 100% tài liệu công ty.
> 2. Đảm bảo tính nhất quán (consistency): nhiều khách hàng hỏi cùng một vấn đề sẽ nhận được câu trả lời đồng nhất, tránh mâu thuẫn hay tranh chấp quyền lợi.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> - Chênh lệch chi phí: GPT-4o (\$0.010/1K) đắt gấp **16.67 lần** GPT-4o-mini (\$0.0006/1K). Với 30.000 lượt/ngày (~10.5 triệu token output), GPT-4o tốn **\$105/ngày** (~$3.150/tháng), trong khi GPT-4o-mini chỉ tốn **\$6.3/ngày** (~$189/tháng).
> - Khi nên dùng GPT-4o: Tác vụ phức tạp đòi hỏi lập luận logic đa bước, phân tích văn bản pháp lý, y tế, hoặc viết thuật toán khó nơi chất lượng là ưu tiên tuyệt đối.
> - Khi nên dùng GPT-4o-mini: Tác vụ phân loại ý định (intent), trích xuất dữ liệu có cấu trúc (JSON), tóm tắt tin nhắn ngắn, hoặc chatbot FAQ phục vụ hàng vạn người dùng mỗi ngày.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> - Giáo viên tiểu học: Câu ngắn, từ ngữ đơn giản, sử dụng hình ảnh ẩn dụ cuốn sổ tay chung của lớp để trẻ 8 tuổi dễ hình dung.
> - Chuyên gia tài chính: Dài hơn, sử dụng nhiều thuật ngữ chuyên sâu (sổ cái phân tán DLT, mật mã học, cơ chế đồng thuận, tính bất biến) và phân tích cơ chế giao dịch.
> - Tác động: System prompt đóng vai trò "chỉ thị định hướng" (steering instruction), thay đổi toàn diện góc nhìn, văn phong, độ sâu kiến thức và đối tượng mục tiêu mà không cần sửa câu hỏi của người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> - Mức chênh lệch thực tế: Đoạn văn tiếng Việt 100 từ có ước lượng `100 / 0.75 ≈ 133` token, nhưng `tiktoken` đếm thực tế khoảng 175 token — chênh lệch thực tế cao hơn ước lượng khoảng **31.5%**.
> - Nguyên nhân: Bộ tokenizer BPE của OpenAI được tối ưu chủ yếu trên ngữ liệu tiếng Anh (đa phần 1 từ = 1 token). Tiếng Việt có hệ thống thanh dấu Unicode UTF-8 đa byte và chiếm tỷ trọng thấp hơn trong tập huấn luyện, khiến mỗi từ thường bị băm thành 1.5 – 2+ subwords.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng (chatbot, trợ lý viết văn/code), giúp giảm thiểu thời gian chờ token đầu tiên (TTFT) để người dùng đọc được ngay câu trả lời mà không cảm thấy ứng dụng bị treo. Ngược lại, non-streaming phù hợp hơn cho các pipeline chạy ngầm giữa các hệ thống (server-to-server), xử lý theo lô (batch jobs), hoặc các tác vụ yêu cầu trích xuất dữ liệu có cấu trúc (JSON Schema) cần có payload hoàn chỉnh trước khi parse.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> - Lợi thế của exponential backoff: Thời gian chờ tăng gấp đôi sau mỗi lần thử (0.1s -> 0.2s -> 0.4s...), giúp nhanh chóng giãn cách tần suất request và tạo khoảng thở để máy chủ giải phóng hàng đợi và phục hồi.
> - Nếu dùng delay cố định: Sẽ gây ra hiện tượng bão thử lại ("thundering herd" / "retry storm"). Hàng nghìn client sẽ cùng lúc đập request trở lại server ở mỗi giây cố định, khiến tình trạng nghẽn nghẽn thêm nghiêm trọng và làm sập hoàn toàn hệ thống.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> - System prompt: `"Bạn là trợ giảng thân thiện của khóa AI, giải thích dễ hiểu, đi thẳng vào trọng tâm và luôn trả lời ngắn gọn bằng tiếng Việt."`
> - Giải thích từ ngữ quan trọng:
>   1. *"Trả lời ngắn gọn, đi thẳng vào trọng tâm":* Giúp người dùng đọc nhanh trên giao diện CLI, đồng thời tiết kiệm số token output (giảm chi phí và độ trễ).
>   2. *"Bằng tiếng Việt":* Giữ ngôn ngữ nhất quán, tránh việc mô hình tự ý chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật.
>   3. *"Trợ giảng thân thiện":* Tạo tâm lý cởi mở, khuyến khích học viên thoải mái đặt câu hỏi mà không sợ bị phán xét.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> - Hạn chế lớn nhất: Lịch sử hội thoại bị cắt cứng ở 3 lượt gần nhất (`history[-6:]`), khiến trợ lý quên sạch các thông tin quan trọng mà người dùng đã đề cập ở các lượt đầu nếu phiên hội thoại kéo dài.
> - Đề xuất cải thiện: Triển khai **Bộ nhớ tóm tắt ngữ cảnh (Conversation Summary Memory)**.
>   - Cách triển khai: Khi `history` vượt quá 3 lượt, gọi mô hình nhỏ giá rẻ (`GPT-4o-mini`) để tóm tắt các lượt cũ thành một đoạn tóm tắt ngắn. Lưu đoạn tóm tắt này vào một system prompt phụ (ví dụ: `{"role": "system", "content": "Tóm tắt ngữ cảnh trước: ..."}`) và gửi kèm 2 lượt chat mới nhất, giúp vừa ghi nhớ dài hạn vừa tiết kiệm token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
