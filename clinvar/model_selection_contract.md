# Contract: Tune siêu tham số, ghi log và chọn mô hình

> **Trạng thái: team đã thống nhất ngày 2026-11-23.** Các quyết định chung được tóm tắt trong `.knowledge/decisions.md`, còn file này là tài liệu tham chiếu chi tiết. Muốn đổi một quy tắc thì cần cả team đồng ý và sửa cả hai file.

## 1. Mục đích và phạm vi

Ba họ mô hình phân loại `CLASS` chọn cấu hình cuối theo cùng một bộ fold, metric, cách lưu cấu hình, định dạng log và quy tắc chọn, để kết quả so sánh được với nhau và tái lập được.

- **Logistic Regression (LR)**, **Random Forest (RF)** và **Gradient Boosting (GB)**: áp dụng toàn bộ mục 2–11. GB dùng thư viện LightGBM; thiết kế riêng của GB nằm trong `.knowledge/gradient_boosting.md`.
- K-means đã được bỏ khỏi dự án (xem `.knowledge/decisions.md`).
- Ngoài phạm vi: thiết kế đặc trưng. LR và RF bắt đầu từ bộ đầu vào Dương đã khóa trong `clinvar/feature_retained/selected.json`. Bộ đầu vào của GB và mọi thay đổi đặc trưng là quyết định riêng (xem §12).

## 2. Dữ liệu và fold

Tuân theo mục Data Contract trong `AGENTS.md`.

- Đầu vào: `clinvar/feature_retained/clean_train.csv`, `y_train.csv`, `fold_assignments.csv`, ghép với nhau qua `source_row`. Đọc đầu vào mô hình bằng `read_model_inputs` trong `study_retained.py`, hàm này áp đúng kiểu dữ liệu của bảng sạch.
- **Fold để chọn cấu hình:** `variant_fold` (1–5). Truyền vào scikit-learn bằng `PredefinedSplit(variant_fold - 1)`. Không tự tạo fold bằng `KFold`/`StratifiedGroupKFold` lúc chạy, vì sẽ ra fold khác.
- **Fold kiểm tra độ bền:** `gene_fold` (1–5), chỉ dùng cho phần chẩn đoán ở §7, không bao giờ dùng để chọn cấu hình.
- Bước điền thiếu, mã hóa và chuẩn hóa nằm chung một `Pipeline` với mô hình, để được fit lại trên phần train của từng fold.
- Fold validation chỉ dùng để chấm điểm. Không dùng nó cho bất kỳ bước fit nào, kể cả early stopping của GB: nếu cần early stopping thì tách tập dừng sớm từ phần train của fold.
- `random_state = 42` cho mọi mô hình và bước lấy mẫu.

## 3. Metric

| Vai trò | Metric | Tính trên |
|---|---|---|
| **Chính (dùng để chọn)** | Average Precision (AP) cho `CLASS = 1`, cùng metric với nghiên cứu đặc trưng của Dương | Fold validation |
| Chẩn đoán overfit | AP trên train, gap = AP train − AP validation | Phần train và fold validation |
| Chẩn đoán calibration | Log loss | Fold validation |
| Chỉ báo cáo, không dùng để chọn | ROC AUC; precision, recall, F1 tại ngưỡng đã chọn (§9) | Fold validation, dự đoán ngoài fold |

Chọn AP làm metric chính vì lớp dương là lớp thiểu số (khoảng 25,2% trên train), và AP tóm tắt được đánh đổi precision–recall mà không phải cố định ngưỡng.

## 4. Quy trình tìm kiếm

1. **Chốt lưới trước khi chạy.** Ghi lưới tham số vào file cấu hình của họ mô hình (§5) và commit trước lần chạy CV đầu tiên. Chỉ được mở rộng lưới theo quy tắc mép ở bước 4; mỗi lần mở rộng là một vòng mới trong file cấu hình và được ghi log.
2. **Mỗi lần chỉ đổi một nhóm tham số.** Tune tham số độ phức tạp chính trước, rồi đến các tham số phụ, giữ cố định lựa chọn của bước trước.
3. **Ngân sách:** tối đa 60 cấu hình cho mỗi họ mô hình, cộng dồn qua mọi vòng. Mỗi cấu hình thử thêm trên cùng 5 fold làm điểm CV của cấu hình thắng bị lạc quan hơn.
4. **Quy tắc mép:** nếu giá trị tốt nhất nằm ở mép lưới, mở rộng lưới thêm một bước về phía đó, tối đa hai lần. Nếu giá trị tốt nhất nằm bên trong lưới thì dừng mở rộng.

**Lưới khởi đầu** (người phụ trách module có thể chỉnh trước lần chạy đầu)

| Mô hình | Thứ tự | Tham số | Giá trị | Thứ tự đơn giản (đơn giản nhất trước) |
|---|---|---|---|---|
| LR | 1 | `C` | 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100 | `C` nhỏ hơn |
| LR | 2 | `penalty` | `l2`, `l1` (solver `saga` hoặc `liblinear`) | `l2`, rồi `l1` |
| LR | 3 | `class_weight` | `None`, `balanced` | `None` |
| RF | 1 | `n_estimators` | Vẽ đường cong 50 → 1.000; lấy giá trị đầu tiên mà sau đó AP validation tăng < 0,002 | Ít cây hơn |
| RF | 2 | `min_samples_leaf` | 1, 2, 5, 10, 20, 50 | Lá lớn hơn |
| RF | 3 | `max_features` | `sqrt`, 0.2, 0.3, 0.5 | Nhỏ hơn |
| RF | 4 | `max_depth` | `None`, 8, 12, 16, 24 | Nông hơn |

Lưới, thứ tự đơn giản và baseline của GB được chốt trong thiết kế ở `.knowledge/gradient_boosting.md`, rồi ghi vào `configs/gradient_boosting.yaml` trước lần chạy CV đầu tiên, theo cùng các quy tắc ở mục này.

Mốc tham chiếu: nghiên cứu đặc trưng của Dương dùng LR `max_iter=3000` với `C=1` mặc định, và RF `n_estimators=80, max_depth=16, min_samples_leaf=5, max_features=6`. Ghi cấu hình này vào mục `baseline` trong file cấu hình và thành dòng `baseline` trong log của mỗi mô hình.

`max_iter` không phải tham số để tune. Nó chỉ cần đủ lớn để solver hội tụ; gặp `ConvergenceWarning` thì tăng `max_iter` trong file cấu hình, không phải dừng sớm.

## 5. Cấu trúc thư mục và file cấu hình

Cấu hình, code mô hình và kết quả nằm ở ba chỗ tách biệt. Mục tiêu là không ai phải sửa code hay notebook để đổi một tham số, và mọi kết quả đều truy ngược được về đúng cấu hình đã sinh ra nó.

**Cấu trúc:** `clinvar/` chỉ giữ dữ liệu và pipeline tiền xử lý hiện có của Dương. Mọi thứ của phần mô hình đặt ở root của repo. Tên họ mô hình theo `snake_case`: `logistic_regression`, `random_forest`, `gradient_boosting`.

| Thư mục (từ root) | Chứa gì | Không được chứa |
|---|---|---|
| `clinvar/` | Dữ liệu gốc, bảng sạch, fold (`feature_retained/`) và các script tiền xử lý của Dương. Phần mô hình chỉ **đọc** từ đây | Code, cấu hình, kết quả của mô hình |
| `configs/` | Một file YAML cho mỗi họ mô hình: `<ho_mo_hinh>.yaml` | Code xử lý, kết quả |
| `src/` | Chỉ code mô hình: dựng pipeline, chạy CV, chọn cấu hình, fit cuối. Mỗi họ mô hình một module, ví dụ `src/logistic_regression.py`. Code đọc mọi tham số từ file cấu hình | Giá trị tham số viết cứng, đường dẫn tuyệt đối, kết quả |
| `results/<ho_mo_hinh>/` | `cv_folds.csv`, `cv_summary.csv`, `test_metrics.csv` (§6, §10) | Cấu hình, code |
| `notebooks/` | Notebook phân tích và vẽ hình. Notebook chỉ đọc file cấu hình, gọi hàm trong `src/` và đọc file trong `results/` | Giá trị tham số viết cứng, vòng tune riêng |
| `tests/` | Test cho `src/` và `configs/`, theo §11 | Dữ liệu lớn, kết quả chạy thật |
| `checkpoints/<ho_mo_hinh>/` | Mô hình đã fit (`.joblib`), sinh lại được từ cấu hình | Đã nằm trong `.gitignore`, không commit |

**Đọc dữ liệu từ `src/`:** các script trong `clinvar/` import lẫn nhau theo kiểu file phẳng (ví dụ `from preprocess import ...`), và các preprocessor `.joblib` đã lưu cần tìm thấy `consequence_encoder.py` khi nạp. Vì vậy `src/` có **một** module đường dẫn chung (ví dụ `src/paths.py`) dựng đường dẫn tới `clinvar/` từ vị trí file của chính nó và thêm `clinvar/` vào đường dẫn import. Các module mô hình chỉ lấy đường dẫn từ module này, không tự ghép đường dẫn.

**Định dạng file cấu hình:** YAML (`.yaml`) cho mọi họ mô hình.

- Lý do: dễ đọc và sửa với cả người không viết code, và chỉ chứa dữ liệu, không chứa logic, nên không ai nhét được code xử lý vào file cấu hình.
- Đọc bằng `yaml.safe_load` (thư viện `pyyaml`). Không dùng `yaml.load` không kèm `Loader` an toàn.
- Thêm `pyyaml` (ghim phiên bản) vào `pyproject.toml`, `uv.lock` và `clinvar/requirements.txt` trong cùng một commit, theo `AGENTS.md`. Thư viện `lightgbm` của GB cũng thêm theo cách này.
- Lưu ý khi viết YAML: `None` của Python viết là `null`; số dạng khoa học phải có cả dấu chấm và dấu của số mũ (`1.0e-4`, `1.0e+2`), vì `pyyaml` đọc `1e-4` hay `1.0e4` thành chuỗi; cách an toàn nhất là viết số thập phân thường (`0.0001`, `100.0`). Code mô hình kiểm tra kiểu của từng tham số sau khi đọc và báo lỗi nếu sai kiểu.

**Nội dung bắt buộc của mỗi file cấu hình**

| Khóa | Nội dung |
|---|---|
| `model` | Tên lớp mô hình, ví dụ `LogisticRegression` |
| `input_set` | Tên bộ đầu vào trong `clinvar/feature_retained/selected.json` (GB: xem §12) |
| `fixed` | Tham số cố định không tune: `random_state = 42`, `max_iter`, `n_jobs`, … |
| `baseline` | Cấu hình mốc ở §4 |
| `rounds` | Danh sách các vòng tìm kiếm. Mỗi vòng gồm tên vòng, lưới tham số và lý do mở vòng (vòng đầu, hoặc quy tắc mép) |
| `simplicity_order` | Thứ tự đơn giản của từng tham số theo §4, để code áp quy tắc chọn ở §8 |
| `selection` | Các ngưỡng của quy tắc chọn ở §8: dung sai, số fold phải thắng |
| `final` | Để trống khi đang tune. Sau khi chọn xong, ghi cấu hình đã chọn và ngưỡng phân lớp (§9), commit **trước** khi chạy test (§10) |

**Quy tắc**

- Không viết cứng giá trị siêu tham số trong `src/` hay notebook. Ngoại lệ duy nhất là `random_state = 42` nếu không đọc được từ cấu hình. Người review PR kiểm tra điểm này.
- Mỗi lần chạy ghi vào log `git_commit` và `config_id` (§6), nên muốn biết một dòng kết quả đến từ cấu hình nào thì xem file cấu hình ở đúng commit đó.
- Sửa lưới đã chạy thì thêm vòng mới, không ghi đè vòng cũ, để log và cấu hình luôn khớp nhau.
- Đường dẫn trong file cấu hình viết tương đối so với root của repo.

## 6. Định dạng log

Mỗi họ mô hình ghi hai file CSV trong `results/<ho_mo_hinh>/`. Các file này nhỏ, được commit cùng phiên bản code và cấu hình đã sinh ra chúng.

**`cv_folds.csv`**: một dòng cho mỗi (cấu hình, cách chia, fold)

| Cột | Nội dung |
|---|---|
| `model` | `LogisticRegression`, `RandomForest`, `LightGBM` |
| `config_id` | Tên ngắn dễ đọc, ví dụ `C=0.01,penalty=l2` |
| `params` | Toàn bộ tham số của mô hình, dạng JSON |
| `round` | Tên vòng tìm kiếm, khớp với `rounds` trong file cấu hình |
| `scheme` | `variant` hoặc `gene` |
| `fold` | 1–5 |
| `n_train`, `n_valid`, `positive_valid` | Số dòng |
| `train_ap`, `valid_ap`, `gap` | Theo §3 |
| `valid_logloss`, `valid_roc_auc` | Theo §3 |
| `n_iter`, `converged` | Số vòng lặp của solver và cờ hội tụ (LR) |
| `max_abs_coef` | Hệ số có trị tuyệt đối lớn nhất (LR) |
| `oob_ap` | AP out-of-bag nếu bật `oob_score=True` (RF) |
| `fit_seconds` | Thời gian chạy `fit` |
| `git_commit` | `git rev-parse --short HEAD` lúc chạy |
| `seed` | 42 |

Cột chẩn đoán riêng của GB (ví dụ số cây thực dùng khi có early stopping) được chốt cùng thiết kế GB và thêm vào bảng này.

**`cv_summary.csv`**: một dòng cho mỗi (cấu hình, cách chia), gồm trung bình, độ lệch chuẩn và sai số chuẩn (SE = std / √5) của từng metric, cộng thêm `folds_up_vs_baseline` (số fold mà AP validation cao hơn dòng `baseline`).

Không bao giờ ghi kết quả trên tập test vào hai file này (xem §10).

## 7. Chẩn đoán overfit

Vẽ **validation curve** cho mỗi tham số ở §4: AP train trung bình và AP validation trung bình theo giá trị tham số, kèm thanh sai số qua 5 fold. Đọc như sau.

| Tín hiệu | Ý nghĩa |
|---|---|
| AP train và AP validation đều thấp và sát nhau | Underfit: cho phép mô hình phức tạp hơn |
| AP validation đạt đỉnh rồi đi ngang hoặc giảm, trong khi AP train vẫn tăng và gap rộng ra | Bắt đầu overfit: dừng ở đỉnh hoặc trước đỉnh |
| Độ lệch chuẩn của AP validation giữa các fold tăng theo độ phức tạp | Mô hình trở nên kém ổn định |
| LR: `max_abs_coef` tăng vọt khi tăng `C` | Hệ số đang khớp theo các mức mã hóa hiếm |
| Log loss validation tăng trong khi AP validation đi ngang | Xác suất dự đoán tự tin quá mức |
| AP trên `gene_fold` giảm mạnh hơn hẳn so với `variant_fold` | Mô hình dựa vào mẫu riêng của từng gene, không chuyển được sang gene chưa gặp |

Lưu ý theo từng mô hình:

- **LR** có ít tham số so với 52.150 dòng, nên gap gần bằng 0 là bình thường, không phải lỗi. Underfit dễ xảy ra hơn overfit.
- **RF** với cây sâu có AP train gần bằng 1, nên gap gần như không nói lên điều gì. Dùng AP validation và AP OOB. Thêm cây không gây overfit mà chỉ làm đường cong phẳng dần, vì vậy `n_estimators` dùng quy tắc đường cong phẳng ở §4.
- **GB** khác RF: mỗi cây mới sửa lỗi của các cây trước, nên thêm cây **có thể** gây overfit. Với `learning_rate` cố định, AP validation thường đạt đỉnh rồi giảm khi tăng số cây, trong khi AP train vẫn tăng. Vì vậy số cây của GB không dùng quy tắc đường cong phẳng của RF, và gap là tín hiệu có ý nghĩa.

## 8. Quy tắc chọn

Áp dụng cho cả ba họ, trên cách chia `variant` trong `cv_summary.csv`.

1. **Điều kiện hợp lệ:** cả 5 fold đều cho AP hợp lệ; với LR, mọi fold đều có `converged = True`.
2. **Tốt nhất:** cấu hình có AP validation trung bình cao nhất.
3. **Dung sai:** tol = max(SE của cấu hình tốt nhất, 0,002). Mức 0,002 là ngưỡng tăng tối thiểu Dương dùng làm mức nhiễu khi chọn đặc trưng (`MIN_GAIN` trong `study_retained.py`).
4. **Chọn cấu hình đơn giản nhất** có AP validation trung bình ≥ tốt nhất − tol, theo `simplicity_order` trong file cấu hình.
5. **Phải hơn baseline:** cấu hình được chọn chỉ thay `baseline` nếu tăng AP validation trung bình ít nhất 0,002 **và** thắng ở ít nhất 4/5 fold. Nếu không, giữ `baseline`.
6. **Ngừng tune** khi giá trị tốt nhất nằm bên trong lưới (quy tắc mép ở §4) và vòng gần nhất làm AP validation trung bình tăng chưa tới 0,002.

Lưu ý: 5 điểm fold dùng chung phần lớn dữ liệu train với nhau, nên SE ở trên đánh giá thấp độ bất định thật. Cần thận trọng với các chênh lệch nhỏ.

## 9. Ngưỡng phân lớp

- Chọn **sau khi** đã chốt siêu tham số, dựa trên xác suất dự đoán ngoài fold trên 52.150 dòng train.
- Tiêu chí: ngưỡng cho F1 cao nhất trên dự đoán ngoài fold, giống `study_endpoints.py` của Dương.
- Ngưỡng được ghi vào mục `final` của file cấu hình cùng cấu hình đã chọn, trước khi dùng tập test.

## 10. Fit cuối và đánh giá test một lần

1. Đóng băng cấu hình và ngưỡng: ghi vào mục `final` của file cấu hình, commit, rồi ghi tóm tắt kèm `git_commit` vào `.knowledge/<module>.md` của module.
2. Fit lại toàn bộ pipeline trên cả 52.150 dòng train; lưu mô hình vào `checkpoints/<ho_mo_hinh>/`.
3. Chấm 13.038 dòng test **đúng một lần**. Ghi kết quả vào `results/<ho_mo_hinh>/test_metrics.csv`, có `git_commit` và cấu hình đã đóng băng.
4. Không thay đổi bất kỳ lựa chọn nào sau khi đã xem kết quả test. Nếu phát hiện lỗi sau đó, sửa lỗi, chạy lại và báo cáo cả hai lần chạy.

**So sánh giữa ba họ:** với mỗi cặp (LR–RF, LR–GB, RF–GB), dùng chênh lệch AP theo từng fold trên cùng các fold `variant_fold` (ΔAP trung bình ± SE, số fold thắng), không chỉ dựa vào điểm test.

## 11. Kiểm thử (`tests/`)

Bổ sung cho mục Testing trong `AGENTS.md`, không thay thế. Mọi quy định trong `AGENTS.md` (mỗi test kiểm một hành vi với giá trị kỳ vọng rõ ràng, không trùng lặp, không đọc nhãn tập test, không sửa, xóa, `skip` hay `xfail` test để cho qua) vẫn áp dụng.

**Vị trí và cách chạy**

- Test của phần mô hình đặt trong `tests/` ở root, song song với `src/`. Test hiện có của Dương vẫn ở `clinvar/test_*.py`, không di chuyển.
- Mỗi họ mô hình một file `tests/test_<ho_mo_hinh>.py`. Phần dùng chung có file riêng: `tests/test_configs.py` (file cấu hình), `tests/test_data_contract.py` (dữ liệu đầu vào).
- Dùng `unittest` như code hiện có để không thêm thư viện. Chạy từ root: `uv run python -m unittest discover -s tests`. Nếu team chuyển sang `pytest` thì thêm vào môi trường theo `AGENTS.md`.
- Toàn bộ `tests/` chạy xong trong khoảng 2 phút, để chạy được trước mỗi lần push.

**Dữ liệu cho test**

- Dùng dữ liệu tổng hợp tạo ngay trong code test, hoặc một mẫu nhỏ lấy từ train (ví dụ 2.000 dòng, lấy bằng `random_state = 42`).
- Không đọc nhãn trong `y_test.csv`. Không chạy bước đánh giá test ở §10 trong test.
- Nếu cần file mẫu cố định thì đặt trong `tests/fixtures/`, mỗi file dưới 1 MB và sinh ra được từ một script.

**Ghi file**

- Test không ghi vào `results/`, `checkpoints/` hay `clinvar/`. Hàm trong `src/` nhận thư mục đầu ra làm tham số, và test truyền vào thư mục tạm.
- Test dùng file cấu hình riêng cho test (lưới nhỏ, 2 fold), tạo trong thư mục tạm hoặc đặt trong `tests/fixtures/`, không sửa file trong `configs/`.

**Các loại test bắt buộc cho mỗi họ mô hình**

| Loại | Kiểm tra gì | Ví dụ |
|---|---|---|
| Unit | Từng phép biến đổi và quy tắc, so với giá trị tính tay trên vài dòng | Quy tắc chọn ở §8 trên một `cv_summary` tổng hợp, chọn đúng cấu hình đơn giản nhất trong dung sai; quy tắc mép ở §4 |
| Chống rò rỉ | Bước điền thiếu, mã hóa, chuẩn hóa và early stopping chỉ dùng phần train của fold | Trung vị học được bằng trung vị của phần train, khác trung vị toàn bộ dữ liệu; GB: tập dừng sớm không chứa dòng nào của fold validation |
| Smoke | Chạy cả quy trình CV trên mẫu nhỏ với cấu hình test | `cv_folds.csv` có đúng các cột ở §6, đúng số dòng (số cấu hình × số fold), không có NaN ở cột metric |
| Tái lập | Chạy hai lần cùng cấu hình cho cùng kết quả | So hai `cv_folds.csv`, bỏ qua `fit_seconds` |

**Test dùng chung**

| File | Kiểm tra gì |
|---|---|
| `tests/test_configs.py` | Mọi file trong `configs/` đọc được bằng `yaml.safe_load`; có đủ khóa bắt buộc ở §5 và không có khóa lạ; tham số đúng kiểu (bắt lỗi `1e-4` bị đọc thành chuỗi); `random_state = 42`; số cấu hình không vượt ngân sách ở §4; `final` hoặc để trống, hoặc có đủ cấu hình và ngưỡng |
| `tests/test_data_contract.py` | `clean_train.csv`, `y_train.csv`, `fold_assignments.csv` ghép được qua `source_row`, đúng 52.150 dòng; train và test không trùng `source_row`; `variant_fold` chỉ gồm 1–5, mỗi fold 10.430 dòng; các cột mà file cấu hình dùng có trong bảng sạch |

Không viết test kiểm tra `src/` có hardcode tham số hay không; điểm này do người review PR kiểm tra (§5).

## 12. Câu hỏi mở

| Câu hỏi | Ai quyết định |
|---|---|
| GB dùng bộ đầu vào nào: bộ RF của Dương, bộ LR, hay một bộ riêng? Mọi thay đổi đặc trưng phải đi qua quy tắc chọn của Dương trên cùng bộ fold | Sơn cùng Dương |
| Lưới, thứ tự đơn giản, baseline và cách xử lý số cây (lưới hay early stopping) của GB | Sơn, ghi trong `.knowledge/gradient_boosting.md` |
| Ai thêm `pyyaml` và `lightgbm` vào môi trường? Việc này sửa cả `clinvar/requirements.txt` của Dương nên cần Dương đồng ý | Người phụ trách module đầu tiên dùng cấu hình, cùng Dương |
| Bộ đầu vào Dương đã khóa là cố định, hay người phụ trách LR/RF được xem lại đặc trưng? Nếu được, mọi thay đổi đặc trưng phải đi qua quy tắc chọn của Dương trên cùng bộ fold | Dương cùng người phụ trách LR/RF |
| Bước `holdout` trong nghiên cứu đặc trưng đã chấm tập test một lần cho cấu hình đã khóa. Lần đánh giá cuối của các mô hình có tính là lần dùng thứ hai không, và báo cáo việc này thế nào? | Cả team |
| Ai phụ trách LR và RF? | Cả team, sau đó ghi vào `.knowledge/architecture.md` |
