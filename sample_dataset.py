labeled = [
    # Shringara
    ("I'm on a date tonight, any romantic songs for the mood?", "Shringara"),
    ("play something that makes me think of my girlfriend", "Shringara"),
    ("need a song for our anniversary dinner", "Shringara"),
    ("something sweet and tender for slow dancing with my wife", "Shringara"),
    ("I've got a huge crush and can't stop thinking about them, suggest some songs", "Shringara"),
    ("I miss him so much, I want love songs", {"Shringara", "Karuna"}),

    # Hasya
    ("give me something silly and fun, I'm in a goofy mood", "Hasya"),
    ("any funny songs? I want to laugh", "Hasya"),
    ("my friends and I want a playlist of comedy songs for the road trip lol", "Hasya"),
    ("I'm in such a playful mood today, what should I listen to?", "Hasya"),
    ("something light and cheerful to crack me up", "Hasya"),
    ("I got the job!! play something to celebrate", {"Hasya", "Veera"}),

    # Karuna
    ("I'm feeling really low today, play something sad", "Karuna"),
    ("my dog passed away yesterday, I just want to cry to some music", "Karuna"),
    ("can you recommend songs for a breakup? I'm heartbroken", "Karuna"),
    ("i feel so lonely tonight", "Karuna"),
    ("I'm having a really bad day, nothing is going right", "Karuna"),
    ("I'm sad, I need something to cheer me up", "Karuna"),

    # Raudra
    ("I'm so angry at my boss right now, need loud angry songs", "Raudra"),
    ("just got cheated on and I'm furious, give me something aggressive", "Raudra"),
    ("I want to scream, play something heavy", "Raudra"),
    ("everything is making me mad today", "Raudra"),
    ("I need a song to vent my rage", "Raudra"),

    # Veera
    ("pump me up, I have a big race tomorrow", "Veera"),
    ("I need a motivational song to push through my workout", "Veera"),
    ("going into a tough interview, give me something to feel brave", "Veera"),
    ("I want an anthem that makes me feel unstoppable", "Veera"),
    ("play something heroic and powerful, epic battle vibes", "Veera"),

    # Bhayanaka
    ("I'm really anxious about my exam tomorrow, can't stop shaking", "Bhayanaka"),
    ("I'm scared to walk home alone tonight", "Bhayanaka"),
    ("I keep having panic attacks and I'm terrified", "Bhayanaka"),
    ("I'm so nervous about the surgery, my heart is racing", "Bhayanaka"),
    ("watching a horror movie later, want something creepy and tense", "Bhayanaka"),

    # Bibhatsa
    ("I'm so disgusted by what I saw on the news today", "Bibhatsa"),
    ("ugh this whole situation is revolting, I feel sick", "Bibhatsa"),
    ("I can't stand how fake and corrupt everyone is, it makes me nauseous", "Bibhatsa"),
    ("I feel gross after that conversation, I need to shake it off", "Bibhatsa"),
    ("found mould all over my kitchen, I'm so grossed out", "Bibhatsa"),

    # Adbhuta
    ("I want a song that is amazing, something that gives me chills", "Adbhuta"),
    ("just saw the northern lights and I'm speechless, what music fits this?", "Adbhuta"),
    ("play something awe-inspiring, like looking at the stars", "Adbhuta"),
    ("I just got back from the Grand Canyon and I'm still in awe", "Adbhuta"),
    ("something mind-blowing and otherworldly please", "Adbhuta"),

    # Shanta
    ("I need to unwind, play something calm and peaceful", "Shanta"),
    ("music for meditation please", "Shanta"),
    ("I want to relax and let my mind go quiet", "Shanta"),
    ("it's a quiet Sunday morning, I want something gentle and soothing", "Shanta"),
    ("help me fall asleep with something slow and calm", "Shanta"),
]
print(len(labeled), "labeled messages")

# Kept for later
# intent_labeled = [
#     ("I'm sad, I need something to cheer me up", "change"),
#     ("play something sad", "match"),
#     ("I'm so angry at my boss right now, need loud angry songs", "match"),
#     ("I'm really anxious about my exam, help me calm down", "change"),
#     ("I just want to cry to some music", "match"),
#     ("I had a rough day, I just want to relax", "change"),
#     ("pump me up, I have a big race tomorrow", "match"),
#     ("I feel lonely tonight, make me feel less alone", "change"),
# ]