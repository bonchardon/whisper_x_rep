from asyncio import run

from loguru import logger

from core.transform_data import AudioTranscription

async def main() -> None:
    transcribe_file = await AudioTranscription().run()
    logger.info(f'The transcription is:\n{transcribe_file}')
    return


if __name__ == '__main__':
    run(main())
