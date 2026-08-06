from deep_translator import GoogleTranslator
from langdetect      import detect as langdetect_detect
from langdetect      import DetectorFactory

DetectorFactory.seed = 0

LANGUAGE_NAMES = {
    "ta"   : "Tamil",
    "hi"   : "Hindi",
    "ar"   : "Arabic",
    "fr"   : "French",
    "de"   : "German",
    "es"   : "Spanish",
    "zh-cn": "Chinese",
    "ja"   : "Japanese",
    "ko"   : "Korean",
    "pt"   : "Portuguese",
    "ru"   : "Russian",
    "it"   : "Italian",
    "en"   : "English"
}


def detect_language(text: str) -> str:
    try:
        return langdetect_detect(text)
    except Exception:
        return 'en'


def translate_to_english(text: str) -> tuple:
    try:
        lang_code = detect_language(text)
        if lang_code == 'en':
            return text, 'en'
        translator = GoogleTranslator(source=lang_code, target='en')
        translated = translator.translate(text)
        return translated or text, lang_code
    except Exception:
        return text, 'en'


def translate_from_english(text: str, target: str) -> str:
    if target in ['en', 'english']:
        return text
    try:
        max_chars = 4500
        if len(text) <= max_chars:
            translator = GoogleTranslator(source='en', target=target)
            return translator.translate(text) or text
        else:
            paragraphs       = text.split('\n')
            translated_parts = []
            current_chunk    = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < max_chars:
                    current_chunk += para + '\n'
                else:
                    if current_chunk.strip():
                        translator = GoogleTranslator(source='en', target=target)
                        translated_parts.append(
                            translator.translate(current_chunk) or current_chunk
                        )
                    current_chunk = para + '\n'
            if current_chunk.strip():
                translator = GoogleTranslator(source='en', target=target)
                translated_parts.append(
                    translator.translate(current_chunk) or current_chunk
                )
            return '\n'.join(translated_parts)
    except Exception:
        return text


def get_language_name(lang_code: str) -> str:
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code.upper())