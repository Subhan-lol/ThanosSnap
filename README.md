# Snap to Dust

Snap your fingers in a Google Meet and crumble into dust that blows away. The spot where you were fills in with the room behind you, and everything else stays your live camera. Snap again and the dust flies back and re-forms you.

## One-time setup

1. **OBS Studio** is installed. You never have to open it. It just provides the **OBS Virtual Camera** that Meet uses.
2. The Python packages are installed on this PC. On another PC, run `pip install -r requirements.txt`.

## Using it

1. Double-click **Start Snap to Dust.bat**, *before* Meet grabs your real webcam.
2. The first time, it builds a picture of your room **without you**. Sit exactly where you'll be in the meeting; the picture is made for *that* view. After the countdown beeps, **lean far left, then far right** (or step out of the shot if the laptop won't move). A small preview shows what it has so far, with the parts it hasn't seen yet in red, plus a % counter. A high double-beep means it's done. If it says **"The laptop moved!"**, lean instead of getting up: a laptop on a bed tips when you stand.
3. In Google Meet go to **Settings → Video → Camera** and pick **OBS Virtual Camera**.
4. **To snap:** raise your hand where the camera sees it and touch your **thumb and middle finger** together. The preview says `hand: READY - snap!`. Then snap. Snap again to come back.

| Key | What it does |
|---|---|
| **F8** (works inside Meet) | snap / un-snap, the silent backup |
| **F9** (works inside Meet) | retake the empty-room photo |
| SPACE / B / M / H / Q (preview window) | snap / retake room / mic on-off / hide the preview's text / quit |

## How it works

- **Seeing the snap:** hand tracking follows 21 points on your hand. A snap means thumb and middle finger together, then the middle finger snapping down into your palm.
- **Hearing the snap:** the mic also listens, as a backup. It only counts if the camera saw your hand in the snap pose just before.
- **Only your spot is replaced:** it cuts out just you (plus your outline and shadow) and fills that area with what's behind you. The rest of the picture stays live.
- **The room picture is lined up with the camera:** it matches distinctive points (furniture edges, wall panel lines, patterns) between the room picture and the live view, then bends the picture to fit exactly. It re-fits whenever the view shifts, at least every 4 seconds, and right when you snap. So if the laptop has tilted a little, the headboard still lines up.
- **Lighting is matched locally:** it compares brightness and colour all around you and carries that smoothly into the gap, so the patch blends in with the wall next to it.
- **Building the room picture:** it stitches together every bit of the room it sees while you move out of the way, lining each frame up first in case the laptop wobbles. Spots seen far from you count more, so your outline and shadow don't get baked in.
- **Changed room detection:** it beeps twice and shows a red banner. **"The camera moved since the room picture"** means the laptop tipped so far that part of the spot where you are was never in the picture; that part would have to be guessed, which shows up as an outline. **"This doesn't look like your room picture"** means you're somewhere else. Either way, press **F9** to retake it from where you are.
- The banner, the text and the mic meter are only in your preview window. Meet never sees them.

## Tips

- If `hand: READY` never appears, bring your hand a bit closer to the camera with your palm facing it. Good lighting helps.
- **Your laptop's mic filters out snap sounds** (Intel Smart Sound noise filtering), so the sound backup will rarely fire. To let snaps through, go to Windows **Settings → System → Sound → Microphone Array** and turn **Audio enhancements** off. Meet has its own noise cancelling, so calls stay clean.

## Options

Add these after `python snap_dust.py`, or to the end of the line in the .bat file:

```
--wind left|right|up    which way the dust blows (default right)
--duration 3            seconds it takes to crumble
--delay 0.6             dramatic pause after the snap
--sensitivity 5         snap-SOUND sensitivity, 1 (strict) to 10 (twitchy)
--no-hand-check         no hand tracking: the snap sound alone triggers it
--no-mic                don't listen to the mic at all
--room photo.jpg        use your own photo of the empty room
--retake                ignore the saved room photo and take a new one
--camera 1              pick a different webcam (see --list-devices)
--mic 3                 pick a different microphone (see --list-devices)
--no-vcam               preview window only (for practising)
--show-hand             draw the tracked hand points in the preview
```

## Troubleshooting

- **"Could not open a webcam"**: Meet (or another app) is using it. Switch Meet to OBS Virtual Camera, then start the app.
- **Black-and-white picture**: you got the IR camera. Use `--camera 1`.
- **An outline of you is left, or the filled-in area doesn't look right**: the laptop has probably moved since the room picture. Sit where you'll be, press F9 and lean left and right instead of getting up. A firm surface under the laptop (a book or tray on the bed) helps a lot.
- **A plain wall with nothing on it** gives it little to line up with. A picture or object in view helps.
