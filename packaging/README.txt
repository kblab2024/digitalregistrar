===========================================================
  Digital Registrar - Pathology Report Annotator
===========================================================

This folder is self-contained. No Python installation needed.


使用方式 / Usage
----------------
1. 將此資料夾整份解壓縮到任意位置 (例如桌面)。
   Extract this folder to any location (e.g. Desktop).

2. 選擇啟動方式 / Choose a launcher:

   * `run` (正式標註 / Production annotation)
     - 資料根目錄 / data root: `workspace\`
     - 標註者清單已鎖定 / annotator list is locked
     - Windows: 雙擊 `run.bat`
     - Unix:     在終端機執行 `./run.sh`

   * `run_demo` (展示模式 / Demo mode)
     - 資料根目錄 / data root: `dummy\`
     - 可於側邊欄新增標註者 / can add new annotators via sidebar
     - Windows: 雙擊 `run_demo.bat`
     - Unix:     在終端機執行 `./run_demo.sh`

   * 第一次執行時 Windows 可能會詢問是否允許防火牆 -
     請選擇「允許存取」(Allow access)。
     On first launch Windows may ask about the firewall -
     choose "Allow access".

3. 瀏覽器會自動開啟 http://localhost:8501。
   The browser opens http://localhost:8501 automatically.

4. 於側邊欄選擇模式 (with_preann / without_preann) 與資料集後開始標註。
   Select the mode (with_preann / without_preann) and dataset
   in the sidebar, then start annotating.

5. 關閉啟動視窗以停止伺服器 (按 Ctrl+C 或直接關閉該視窗)。
   Close the launcher window to stop the server
   (Ctrl+C, or just close the window).


資料放置 / Place your dataset
-----------------------------
將您的資料依下列結構放入 `workspace\` 資料夾：
Place your data inside `workspace\` using this structure:

   workspace\
     with_preann\
       data\
         <dataset>\
           reports\<organ>\<case_id>.txt
           preannotation\<model>\<organ>\<case_id>.json
           annotations\<annotator>\<organ>\<case_id>.json
     without_preann\
       data\
         <dataset>\
           reports\<organ>\<case_id>.txt
           annotations\<annotator>\<organ>\<case_id>.json

可參考同層 `dummy\` 資料夾的範例檔案格式。
See the sibling `dummy\` folder for example file formats.

您的標註會直接寫入上述 annotations 資料夾，可隨時備份。
Your saved annotations write directly into the `annotations/`
folders above; back them up whenever you like.


疑難排解 / Troubleshooting
-------------------------
* 瀏覽器沒有自動開啟:
  手動前往 http://localhost:8501
  Browser did not open automatically:
  manually navigate to http://localhost:8501.

* 8501 連接埠被其他程式佔用:
  編輯啟動檔案 (run.bat / run.sh), 將 `--server.port=8501`
  改成其他數字 (例如 8502), 再重新執行。
  Port 8501 is already in use:
  edit the launcher file (run.bat / run.sh), change
  `--server.port=8501` to e.g. 8502, then run again.

* `run.bat` 視窗一閃而過看不到錯誤:
  打開「命令提示字元」(Command Prompt), 把 run.bat 拖進去
  執行, 即可看到錯誤訊息。
  The `run.bat` window flashes and closes without showing
  an error: open Command Prompt and drag `run.bat` into it
  to keep the window open.


系統需求 / Requirements
----------------------
* Windows 10 / 11 (64-bit)  或  Linux x86_64
* ~200 MB disk space (Python and all dependencies are bundled)
* No Python installation needed.
