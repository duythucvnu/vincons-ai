# vincons-ai

Concise deployment guide to run and integrate the training pipeline.

## Directory Structure

```text
vincons-ai/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py                # YOLO26s training script
├── telebot_runner.py       # Telegram bot runner
├── dataset/                # Place raw images & labels here
└── runs/                   # Output weights
```

## Setup

```bash
git clone https://github.com/duythucvnu/vincons-ai.git
cd vincons-ai
pip3 install pyTelegramBotAPI
nohup python3 telebot_runner.py > bot.log 2>&1 &
```
