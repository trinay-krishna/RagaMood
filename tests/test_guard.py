import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chatbot.guard import asks_question, has_feeling, looks_like_song_list


def test_song_list_detection():
    assert looks_like_song_list('1. "Kalyani Kriti" - Kalyani - Sanjay\n2. "Thillana" - Kafi - L. Subramaniam')
    assert looks_like_song_list('"Kalyani" - "Kalyani" - "Sanjay"\n"Kharaharapriya" - "Kharaharapriya" - "T. M. Krishna"')
    assert not looks_like_song_list("You're welcome! Enjoy the music.")
    assert not looks_like_song_list('Here is one: 1. "Kalyani Kriti" - Kalyani - Sanjay')     # a single item is not a list
    assert not looks_like_song_list("Carnatic music has kritis - compositions - and varnams.")


def test_question_and_feeling():
    assert asks_question("How are you feeling?") and not asks_question("Enjoy!")
    assert has_feeling("I can't stop smiling") and has_feeling("I'm furious at my boss")
    assert not has_feeling("raga Kalyani") and not has_feeling("thanks!") and not has_feeling("play something")
