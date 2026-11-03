"""Labelled test messages for the chat agent (author-written; review the labels before quoting numbers).

Each case: id, category, turns (user messages, the last one is scored), gold_tool (acceptable tools),
gold_rasas (acceptable rasas for the rewritten mood), gold_entity (any of these substrings must appear
in the rewritten keyword query).  gold_tool "none" = answer without a tool, "clarify" = ask a question.
"""
MOOD, CAT, MIX = "recommend_by_mood", "search_catalog", "recommend_mixed"
UP = ["Hasya", "Shringara", "Veera"]       # what "cheer me up" may reasonably mean

_cases = []


def _add(category, turns, tool, rasas=(), entity=()):
    _cases.append({"id": f"{category}-{len(_cases):03d}", "category": category,
                   "turns": [turns] if isinstance(turns, str) else list(turns),
                   "gold_tool": [tool] if isinstance(tool, str) else list(tool),
                   "gold_rasas": list(rasas), "gold_entity": list(entity)})


for text, rasas in [
    ("I'm feeling really sad today", ["Karuna"]),
    ("I miss my mother so much", ["Karuna"]),
    ("My heart is broken", ["Karuna"]),
    ("I just lost my grandfather and I feel empty", ["Karuna"]),
    ("I'm so happy today!", ["Hasya", "Shringara"]),
    ("I can't stop smiling", ["Hasya", "Shringara"]),
    ("Everything is going great, I feel joyful", ["Hasya", "Shringara"]),
    ("I'm in a playful, joking mood", ["Hasya"]),
    ("I want to relax and feel calm", ["Shanta"]),
    ("I need some peace and quiet", ["Shanta"]),
    ("I'm feeling serene and want to meditate", ["Shanta"]),
    ("I'm in love", ["Shringara"]),
    ("I miss my partner and feel tender", ["Shringara"]),
    ("Feeling romantic tonight", ["Shringara"]),
    ("I need courage for my exam tomorrow", ["Veera"]),
    ("I feel determined and powerful", ["Veera"]),
    ("Pump me up, I have a big match", ["Veera"]),
    ("I'm furious at my boss", ["Raudra"]),
    ("I feel so angry I can't think straight", ["Raudra"]),
    ("I'm terrified of tomorrow's results", ["Bhayanaka"]),
    ("I feel anxious and scared", ["Bhayanaka"]),
    ("I'm amazed by the night sky", ["Adbhuta"]),
    ("That performance was awe-inspiring", ["Adbhuta"]),
    ("I'm disgusted by what I saw today", ["Bibhatsa"]),
    ("I'm sick of all this corruption", ["Bibhatsa"]),
]:
    _add("mood", text, MOOD, rasas)

for text, rasas in [
    ("I'm sad, find something to cheer me up", UP),
    ("I feel down, I need something uplifting", UP),
    ("I'm angry, calm me down", ["Shanta"]),
    ("I'm so stressed, help me unwind", ["Shanta"]),
    ("I'm anxious, play something soothing", ["Shanta"]),
    ("I'm exhausted and unmotivated, give me some energy", ["Veera"]),
    ("I'm bored, something exciting please", ["Veera", "Adbhuta", "Hasya"]),
    ("I'm lonely, make me feel loved", ["Shringara"]),
    ("I'm scared, help me feel brave", ["Veera"]),
    ("I can't sleep because I'm worried, something to settle me", ["Shanta"]),
    ("I'm sad and I want a song that matches how I feel", ["Karuna"]),
    ("I'm down, I want something sad to cry to", ["Karuna"]),
]:
    _add("inversion", text, MOOD, rasas)

for text, ent in [
    ("Sanjay Subrahmanyan kriti", ["sanjay"]),
    ("songs by Neyveli B Venkatesh", ["neyveli"]),
    ("play Ajoy Chakrabarty", ["ajoy"]),
    ("Akkarai Subhalakshmi", ["akkarai"]),
    ("raga Kalyani", ["kalyani"]),
    ("raag Bhairavi", ["bhairavi"]),
    ("play Thodi", ["thodi", "todi"]),
    ("find Kamas", ["kamas"]),
    ("Behag songs", ["behag"]),
    ("Shankarabharanam kriti", ["shankarabharanam", "sankarabharanam"]),
    ("any thillana?", ["thillana"]),
    ("a varnam please", ["varnam"]),
    ("khayal by Ajoy Chakrabarty", ["ajoy"]),
    ("thumri recordings", ["thumri"]),
    ("Hindustani khayal in raga Bageshri", ["bageshri", "bageshree"]),
    ("slow khayal in teentaal", ["khayal", "teentaal"]),
    ("Carnatic kriti in Mohana", ["mohana"]),
    ("Sanjay Subrahmanyan Thodi", ["sanjay", "thodi", "todi"]),
    ("do you have Saurashtram?", ["saurashtram"]),
    ("Mangalam pieces", ["mangalam"]),
    ("songs in raga Sri", ["sri"]),
    ("ragamalika", ["ragamalika"]),
    ("Kaustuv Kanti Ganguli", ["kaustuv"]),
    ("Paromita Mukherjee khayal", ["paromita"]),
]:
    _add("keyword", text, CAT, entity=ent)

for text, rasas, ent in [
    ("I'm sad, play raga Ahibhairav", ["Karuna"], ["ahibhairav", "ahir"]),
    ("Feeling happy, something by Sanjay Subrahmanyan", ["Hasya", "Shringara"], ["sanjay"]),
    ("I'm in love, Kalyani please", ["Shringara"], ["kalyani"]),
    ("I'm heartbroken, a Bageshri khayal", ["Karuna"], ["bageshri", "bageshree"]),
    ("I feel calm, Thodi by Neyveli B Venkatesh", ["Shanta"], ["thodi", "todi", "neyveli"]),
    ("I need courage, play a thillana", ["Veera"], ["thillana"]),
    ("I'm angry, something by Ajoy Chakrabarty to calm me down", ["Shanta"], ["ajoy"]),
    ("Cheer me up with raga Mohana", UP, ["mohana"]),
    ("I miss home, any Kamas?", ["Karuna"], ["kamas"]),
    ("I'm joyful, a kriti in Shankarabharanam", ["Hasya", "Shringara"], ["shankarabharanam", "sankarabharanam"]),
    ("Feeling peaceful, Akkarai Subhalakshmi please", ["Shanta"], ["akkarai"]),
    ("I'm sad, a thumri would be nice", ["Karuna"], ["thumri"]),
    ("I'm amazed and I want Desh", ["Adbhuta"], ["desh"]),
    ("Stressed out, play Bhairavi to unwind", ["Shanta"], ["bhairavi"]),
    ("I'm romantic tonight, Behag", ["Shringara"], ["behag"]),
    ("Feeling brave, Meghamalhaar", ["Veera"], ["megh"]),
]:
    _add("mixed", text, MIX, rasas, ent)

for turns, tool, rasas, ent in [
    (["I'm feeling sad today", "something more upbeat now"], MOOD, UP, []),
    (["I'm happy today", "actually make it calmer"], MOOD, ["Shanta"], []),
    (["I need to relax", "more like that"], MOOD, ["Shanta"], []),
    (["play raga Kalyani", "same raga but a different artist"], [CAT, MIX], [], ["kalyani"]),
    (["Sanjay Subrahmanyan songs", "now something by Ajoy Chakrabarty"], CAT, [], ["ajoy"]),
    (["I'm sad", "now play raga Bageshri for that"], MIX, ["Karuna"], ["bageshri", "bageshree"]),
    (["raga Thodi", "I'm feeling low, anything with it?"], [MIX, CAT], ["Karuna"], ["thodi", "todi"]),
    (["I feel angry", "calm me down instead"], MOOD, ["Shanta"], []),
    (["find a thillana", "something slower"], [CAT, MIX], [], ["thillana"]),
    (["I'm in love", "a Hindustani one please"], [MIX, MOOD], ["Shringara"], []),
    (["play Behag", "thanks!"], "none", [], []),
    (["I'm sad", "thanks, that helps"], "none", [], []),
    (["I'm scared", "I meant I want to feel brave"], MOOD, ["Veera"], []),
    (["Kriti by Neyveli B Venkatesh", "more"], CAT, [], ["neyveli"]),
    (["I feel joyful", "not that, something more romantic"], MOOD, ["Shringara"], []),
    (["play something", "I'm feeling nostalgic and sad"], MOOD, ["Karuna"], []),
    (["hello", "I want a calm song"], MOOD, ["Shanta"], []),
    (["I'm sad, find a Bageshri", "now the same raga but happy"], MIX, ["Hasya", "Shringara"], ["bageshri", "bageshree"]),
]:
    _add("followup", turns, tool, rasas, ent)

for text in ["hi there", "thank you!", "what can you do?", "who are you?", "what is a raga?",
             "tell me a joke", "what's the weather like?", "what is the difference between Carnatic and Hindustani music?"]:
    _add("chitchat", text, "none")

for text in ["play something", "suggest a song", "recommend me some music", "I want some music",
             "surprise me", "give me a good song"]:
    _add("vague", text, "clarify")

CASES = _cases


def load_cases():
    return list(CASES)
