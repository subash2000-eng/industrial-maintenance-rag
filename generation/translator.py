"""
Multilingual Support — Translation Module
Fixed version — uses correct deep-translator API.
"""

from deep_translator import GoogleTranslator
from langdetect      import detect as langdetect_detect
from langdetect      import DetectorFactory

# Make language detection consistent
DetectorFactory.seed = 0

# Language code to name mapping
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
    """
    Detect the language of input text.
    Returns language code like 'ta', 'hi', 'en'.
    """
    try:
        lang = langdetect_detect(text)
        return lang
    except Exception:
        return 'en'


def translate_to_english(text: str) -> tuple:
    """
    Translate any language text to English.

    Returns:
        (translated_text, detected_language_code)
    """
    try:
        # First detect the language
        lang_code = detect_language(text)

        # If already English return as is
        if lang_code == 'en':
            return text, 'en'

        # Translate to English
        translator = GoogleTranslator(
            source=lang_code,
            target='en'
        )
        translated = translator.translate(text)
        return translated or text, lang_code

    except Exception as e:
        print(f"  ⚠️  Translation error: {e}")
        return text, 'en'


def translate_from_english(
    text  : str,
    target: str
) -> str:
    """
    Translate English text to target language.

    Args:
        text  : English text to translate
        target: Target language code ('ta', 'hi', etc.)

    Returns:
        Translated text
    """
    if target in ['en', 'english']:
        return text

    try:
        max_chars = 4500

        if len(text) <= max_chars:
            translator = GoogleTranslator(
                source='en',
                target=target
            )
            result = translator.translate(text)
            return result or text

        else:
            # Split long text into paragraphs
            paragraphs       = text.split('\n')
            translated_parts = []
            current_chunk    = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) < max_chars:
                    current_chunk += para + '\n'
                else:
                    if current_chunk.strip():
                        translator = GoogleTranslator(
                            source='en',
                            target=target
                        )
                        translated_parts.append(
                            translator.translate(
                                current_chunk
                            ) or current_chunk
                        )
                    current_chunk = para + '\n'

            if current_chunk.strip():
                translator = GoogleTranslator(
                    source='en',
                    target=target
                )
                translated_parts.append(
                    translator.translate(
                        current_chunk
                    ) or current_chunk
                )

            return '\n'.join(translated_parts)

    except Exception as e:
        print(f"  ⚠️  Translation error: {e}")
        return text


def get_language_name(lang_code: str) -> str:
    """Convert language code to readable name."""
    return LANGUAGE_NAMES.get(
        lang_code.lower(),
        lang_code.upper()
    )


# ── Test ─────────────────────────────────────────────────
if __name__ == "__main__":

    print("🌐 Testing Multilingual Support\n")

    test_queries = [
        "மோட்டார் தாங்கி அதிகமாக சூடாகிறது",
        "मोटर बेयरिंग गर्म हो रही है",
        "Motor bearing overheating",
        "Le roulement du moteur surchauffe"
    ]

    for query in test_queries:
        print(f"Input: {query}")

        en_text, detected = translate_to_english(query)
        lang_name         = get_language_name(detected)

        print(f"  Detected : {lang_name} ({detected})")
        print(f"  English  : {en_text}")

        if detected != 'en':
            back = translate_from_english(en_text, detected)
            print(f"  Back     : {back}")

        print()

    print("✅ Test complete")