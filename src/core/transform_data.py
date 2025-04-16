from math import ceil
from json import dump
from os import environ

from openai import AsyncOpenAI

from pydub import AudioSegment

from loguru import logger
from dotenv import load_dotenv

from src.core import consts

load_dotenv()


class AudioTranscription:

    def __init__(self):
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=environ.get('OPENAI_AI_KEY'))
        self.audio_file: str = 'data/vol_engineers.wav'

    async def audio_chunks(self):
        if not (audio := AudioSegment.from_file(self.audio_file)):
            logger.warning('There is an issue when opening a file.')
        logger.info(len(audio))
        total_length: int = len(audio)
        num_segments: int = ceil(total_length / consts.SEGMENT_LENGHT)
        for i in range(num_segments):
            start_time = i * consts.SEGMENT_LENGHT
            end_time = min((i + 1) * consts.SEGMENT_LENGHT,
                           total_length)
            segment = audio[start_time:end_time]
            output_file = f'data/chunks/vol_{consts.FILE_SAVE}_part_{i + 1}.wav'
            segment.export(output_file, format='wav')
            logger.info(f'Exported: {output_file}')

    @staticmethod
    async def audio_chunk_detection():
        try:
            return [consts.AUDIO_FILEs_PATH.format(num=num) for num in range(1, 129)]
        except Exception as exc:
            logger.error(exc)

    async def trans_audio(self):
        file: str = self.audio_file
        logger.info(f'Working on this file {file}!')
        if not (audio_file := open(file, 'rb')):
            logger.error('There is an error when opening the file!')
        transcription: str = await self.client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file,
            response_format='text',
            prompt='Ти повинен транскрибувати текст українською мовою. '
                   'Як відповідь має бути текст у JSON форматі, ',
                   # 'Provide a transcription with identifications of a speaker, i.e.: Speaker_1, Speaker_2, etc.',
            language='uk',
            # diarization=True
        )
        logger.info(transcription)
        with open('core/transcribed_file/vol_engineers.json', 'w', encoding='utf-8') as f:
            dump(transcription, f, ensure_ascii=False, indent=4, )
        return transcription

    async def run(self) -> None:
        await self.audio_chunks()
        # await self.trans_audio()
