import os
import re
import subprocess
import threading
import telebot

BOT_TOKEN = "8854012908:AAEoj53-TfProCaTB8WbtfSP6eDW27Ehsq0"
ALLOWED_CHAT_ID = 6361313344
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

bot = telebot.TeleBot(BOT_TOKEN)


def run_training_pipeline():
    try:
        os.chdir(PROJECT_DIR)

        git_output = subprocess.check_output(
            ["git", "pull"],
            text=True,
            stderr=subprocess.STDOUT,
        )

        bot.send_message(
            ALLOWED_CHAT_ID,
            f"*Source code updated:*\n`{git_output}`",
            parse_mode="Markdown",
        )

        cmd = "docker compose up --build"

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        bot.send_message(
            ALLOWED_CHAT_ID,
            "*Starting Docker and building the environment...*",
            parse_mode="Markdown",
        )

        last_epoch_sent = -1
        has_sent_device_info = False

        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)

            line_str = line.strip()

            if "SYS_DEVICE_DETECTED:" in line_str:
                device = line_str.split("SYS_DEVICE_DETECTED:")[1].strip()
                bot.send_message(
                    ALLOWED_CHAT_ID,
                    f"*Detected hardware:* `{device}`",
                    parse_mode="Markdown",
                )
                has_sent_device_info = True

            epoch_match = re.search(r"\b(\d+)/100\b", line_str)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                if epoch_num != last_epoch_sent:
                    bot.send_message(
                        ALLOWED_CHAT_ID,
                        f"*Training Epoch:* `{epoch_num}/100`",
                        parse_mode="Markdown",
                    )
                    last_epoch_sent = epoch_num

            parts = line_str.split()

            if len(parts) >= 7 and parts[0] == "all":
                try:
                    int(parts[1])
                    int(parts[2])

                    precision = float(parts[3])
                    recall = float(parts[4])
                    map50 = float(parts[5])
                    map50_95 = float(parts[6])

                    if (
                        0 <= precision <= 1.05
                        and 0 <= recall <= 1.05
                        and 0 <= map50 <= 1.05
                    ):
                        summary_msg = (
                            f"*Validation Metrics (Epoch {last_epoch_sent}/100):*\n"
                            f"- Precision: `{precision:.4f}`\n"
                            f"- Recall: `{recall:.4f}`\n"
                            f"- mAP50: `{map50:.4f}`\n"
                            f"- mAP50-95: `{map50_95:.4f}`"
                        )

                        bot.send_message(
                            ALLOWED_CHAT_ID,
                            summary_msg,
                            parse_mode="Markdown",
                        )

                except ValueError:
                    pass

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            bot.send_message(
                ALLOWED_CHAT_ID,
                "*Training completed successfully. YOLO26s weights have been saved to `/runs`.*",
                parse_mode="Markdown",
            )
        else:
            bot.send_message(
                ALLOWED_CHAT_ID,
                f"*Training failed.* Exit code: `{return_code}`",
                parse_mode="Markdown",
            )

    except Exception as e:
        bot.send_message(
            ALLOWED_CHAT_ID,
            f"*System error:* `{str(e)}`",
            parse_mode="Markdown",
        )


@bot.message_handler(commands=["train"])
def handle_train(message):
    if message.chat.id != ALLOWED_CHAT_ID:
        bot.reply_to(message, "You are not authorized to use this bot.")
        return

    thread = threading.Thread(target=run_training_pipeline)
    thread.start()


print("Bot is listening for commands...")

bot.infinity_polling()