import logging

import cv2
import numpy as np
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler

from model import get_face_neighbours

with open('token') as f:
    TOKEN = f.read().strip()

WELCOME_MESSAGE = 'Welcome to K-pop alike bot! Send a selfie to begin.'

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.WARN)

logger = logging.getLogger(__name__)

updater = Updater(token=TOKEN)
dp = updater.dispatcher


def start(bot, update):
    logger.info("Bot Started")
    logger.info("Token: %s" % TOKEN)
    update.message.reply_text(WELCOME_MESSAGE)


def help(bot, update):
    update.message.reply_text('Send a selfie to begin!')


def error(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))


def get_neighbours(bot, update):
    image_bytes = update.message.photo[-1].get_file().download_as_bytearray()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    neighbours = get_face_neighbours(image)
    chat_id = update.message.chat_id
    for neighbour in neighbours:
        bot.send_message(chat_id=chat_id, text=neighbour[0])
        bot.send_photo(chat_id=chat_id, photo=open(neighbour[1], 'rb'))


def main():
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    dp.add_handler(MessageHandler(Filters.text, help))
    dp.add_handler(MessageHandler(Filters.photo, get_neighbours))

    dp.add_error_handler(error)
    updater.start_polling(clean=True)
    updater.idle()


if __name__ == '__main__':
    main()
