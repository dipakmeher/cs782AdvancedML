import os

def log_txt_lengths(input_folder, log_path):
    with open(log_path, "w") as log:
        log.write("===== TXT FILE LENGTHS =====\n")

        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        length = len(content)   # character count
                        # If you want word count: length = len(content.split())

                    log.write(f"{filename}: {length}\n")
                except Exception as e:
                    log.write(f"{filename}: ERROR - {e}\n")

        log.write("===== DONE =====\n")


# ---------- USAGE ----------
input_folder = "/projects/cdomenic/HS_CINA/Dipak/GraphRAGProject/cs782Advancedml/linkkg-no-str-prompt/input/"
log_path = os.path.join(input_folder, "log.txt")

log_txt_lengths(input_folder, log_path)

print("Done. Check log.txt")
