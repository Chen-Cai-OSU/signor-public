import argparse
import os

from signor.configs.util import dict2arg
from signor.ioio.dir import home_dir
from signor.utils.system import detect_sys


def email():
    import smtplib

    content = ("Content to send")

    mail = smtplib.SMTP('smtp.gmail.com', 587)

    mail.ehlo()

    mail.starttls()

    mail.login('chencai.math@gmail.com', 'password')

    mail.sendmail('chencai.math@gmail.com', 'destination_email@gmail.com', content)

    mail.close()

    print("Sent")

from signor.format.format import timestamp, red
import slacker


def telegram():
    import telegram
    bot = telegram.Bot(token='1651053843:AAG9-OHILPS5DRzpctG7sKBzf_qNxkBaeRo')

    bot.send_message(chat_id='1651053843',
                     text="*bold* _italic_ `fixed width font` [link](http://google.com)\.",
                     parse_mode=telegram.ParseMode.MARKDOWN_V2)
    # print(bot.get_me())

def tele():
    # importing all required libraries
    import telebot
    from telethon.sync import TelegramClient
    from telethon.tl.types import InputPeerUser, InputPeerChannel
    from telethon import TelegramClient, sync, events

    # get your api_id, api_hash, token
    # from telegram as described above
    api_id = 'API_id'
    api_hash = 'API_hash'
    token = 'bot token'

    # your phone number
    phone = 'YOUR_PHONE_NUMBER_WTH_COUNTRY_CODE'

    # creating a telegram session and assigning
    # it to a variable client
    client = TelegramClient('session', api_id, api_hash)

    # connecting and building the session
    client.connect()

    # in case of script ran first time it will
    # ask either to input token or otp sent to
    # number or sent or your telegram id
    if not client.is_user_authorized():
        client.send_code_request(phone)

        # signing in the client
        client.sign_in(phone, input('Enter the code: '))

    try:
        # receiver user_id and access_hash, use
        # my user_id and access_hash for reference
        receiver = InputPeerUser('user_id', 'user_hash')

        # sending message using telegram client
        client.send_message(receiver, message, parse_mode='html')
    except Exception as e:

        # there may be many error coming in while like peer
        # error, wwrong access_hash, flood_error, etc
        print(e);

        # disconnecting the telegram session
    client.disconnect()


def slack(program='xyz', args=None, status='Done'):
    return
    # todo: slacker.Error: method_deprecated
    assert detect_sys()=='Linux'
    if 'cai.507' in home_dir():
        machine = 'osu'
    elif 'chen' in home_dir():
        machine = 'ucsd'
    else:
        raise NotImplementedError

    slack_cli = '~/anaconda3/bin/slack-cli '

    if args:
        s = dict2arg(vars(args))
        flag =  f'with args:{s} '
    else:
        flag = ''
    cmd = f'{slack_cli} -d slackbot "{machine} [{timestamp()}].\n **{status}** program {program}.\n{flag} " '

    try:
        os.system(cmd)
        # os.system(f'{slack_cli} -d slackbot "Hello"')
    except slacker.Error:
        print(red('slacker error'))
    except:
        print('Exception (likely requests.exceptions.HTTPError)')

    print(cmd)

parser = argparse.ArgumentParser(description='cnn baseline')
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--n_epoch', type=int, default=200, help='')

if __name__ == '__main__':
    # telegram()
    # exit()
    slack(args=parser.parse_args())
    print(1 / 1.0)
    exit()
    from time import sleep

    sleep(5)
    slack()
