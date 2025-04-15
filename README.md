# **Cognito Canvas** ✨🧠

## DESCRIPTION

Alright, geniuses, listen up. Cognito Canvas is your personal digital whiteboard that's actually smart. Scribble, draw, jot down equations – this Kivy app takes your chaotic brilliance and turns it into clean code, math solutions, and summarized notes. Think J.A.R.V.I.S., but for your doodles. Runs on Android, because who carries a laptop anymore?

## FEATURES

*   **Interactive Kivy Canvas:** Go wild. Draw shapes, write text, sketch flowcharts. It's your playground. ✍️
*   **Handwritten Math Solver:** Scrawled an equation? Our OCR reads it (yeah, even *your* handwriting) and SymPy solves it. Basic algebra/arithmetic, don't get crazy. 🧮
*   **Flowchart-to-Code Stubs:** Draws basic flowcharts (shapes, arrows)? OpenCV spots 'em and spits out Python function/conditional stubs. It's like magic, but it's just code. 🪄
*   **AI Note Taker:** Reads your text (OCR again) and eyeballs your drawings to give you the TL;DR. Big brain summaries, low effort. 😎
*   **Multi-modal Input Processing:** It looks at your drawings *and* reads your text to actually get the vibe. Context is everything, peeps.

## LEARNING BENEFITS

Level up your game. You'll build a multi-modal AI, smash Computer Vision (OpenCV) with OCR (EasyOCR) and math vibes (SymPy). Get hands-on with Kivy for mobile apps and UI/UX that doesn't suck. Basically, build your own mini-J.A.R.V.I.S. and prove you've got a heart... for killer productivity. 🚀

## TECHNOLOGIES USED

The Stark Tech behind the magic:

*   **kivy:** The suit – framework and UI for the Android app.
*   **opencv-python:** The eyes – image processing, shape spotting, prepping for OCR.
*   **easyocr:** The translator – reads handwriting and text like a champ.
*   **sympy:** The brainiac – parses and solves those math problems.
*   **networkx (Optional):** For when your flowcharts get spicy and need real analysis.
*   **nltk or spacy (Optional):** If you wanna get fancy with text summaries.

## SETUP AND INSTALLATION

Less talking, more doing. Get this running:

```bash
git clone https://github.com/Omdeepb69/cognito-canvas.git
cd cognito-canvas
pip install -r requirements.txt
```

Boom. Done. You're welcome.

## USAGE

Just run the main script, `main.py`. Easy peasy. Tweak `config.json` if you feel like messing with the settings (at your own risk, obviously).

## PROJECT STRUCTURE

It's organized, unlike your desk probably.

*   `src/`: Where the core magic happens (source code).
*   `tests/`: Making sure things don't blow up (unit tests).
*   `docs/`: Manuals are for rookies, but here's some info anyway (documentation).

## LICENSE

MIT License. Means you can play with it, just don't sue me if your doodles accidentally summon a demon or something. Share nicely.