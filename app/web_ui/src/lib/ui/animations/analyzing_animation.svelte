<script lang="ts">
  // Generic "analyzing" animation: a list of items appearing, getting marked
  // as pass/fail, and flying out. Originally written for the spec builder; the
  // copy below is now parameterized so other flows (e.g. data guide preview
  // generation) can reuse the visual.
  import Warning from "$lib/ui/warning.svelte"
  export let title: string
  export let description: string
  export let warning: string | null = null
</script>

<div class="flex flex-col items-center justify-center">
  <div class="flex flex-col max-h-[250px] max-w-[250px] mt-6">
    <svg
      width="250"
      height="250"
      viewBox="0 0 500 500"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <clipPath id="canvas-clip">
          <rect width="500" height="500" />
        </clipPath>
      </defs>
      <rect width="500" height="500" fill="white" />
      <g clip-path="url(#canvas-clip)">
        <!--
      ╔══════════════════════════════════════════════════════════════════════════════╗
      ║                         ANIMATION TIMELINE DOCUMENTATION                     ║
      ╚══════════════════════════════════════════════════════════════════════════════╝
      
      OVERVIEW:
      This animation shows 8 boxes appearing with staggered timing, followed by 
      checkmarks and X icons appearing on them, then all elements fly out to the right.
      
      TIMELINE STRUCTURE:
      ────────────────────────────────────────────────────────────────────────────────
      
      PHASE 1: STAGGERED APPEARANCE (0.1s - 1.45s)
        • All 8 boxes appear at staggered times with scale animations
        • Each box: scale animation (0.3s duration) + opacity fade in (0.1s)
        • Box appearance times (top to bottom, 0.15s intervals):
          - Box 1: 0.10s
          - Box 2: 0.25s
          - Box 3: 0.40s
          - Box 4: 0.55s
          - Box 5: 0.70s
          - Box 6: 0.85s
          - Box 7: 1.00s
          - Box 8: 1.15s (finishes at 1.45s with 0.3s animation)
      
      PHASE 2: ICONS APPEAR (1.7s - 3.1s)
        • Checkmarks and X icons appear on all 8 boxes
        • Start time: 1.7s (0.25s buffer after appearance completes)
        • Staggered appearance times (0.2s intervals):
          - Checkmark 1 (Box 1): 1.7s
          - Checkmark 2 (Box 2): 1.9s
          - X icon 1 (Box 3):    2.1s
          - Checkmark 3 (Box 4): 2.3s
          - Checkmark 4 (Box 5): 2.5s
          - Checkmark 5 (Box 6): 2.7s
          - X icon 2 (Box 7):    2.9s
          - X icon 3 (Box 8):    3.1s
      
      PHASE 3: FLY OUT LEFT/RIGHT (4.0s - 5.5s)
        • After 0.9s hold, all boxes and icons fly out
        • Checkmark boxes: fly out to the RIGHT (x=600)
        • X icon boxes: fly out to the LEFT (x=-400)
        • Start time: 4.0s (0.9s hold after last icon appears at 3.1s)
        • Duration: 0.6s per row
        • Staggered by 0.1s (top to bottom):
          - Row 1 (Box 1 + Check 1): 4.0s - 4.6s → RIGHT
          - Row 2 (Box 2 + Check 2): 4.1s - 4.7s → RIGHT
          - Row 3 (Box 3 + X 1):     4.2s - 4.8s → LEFT
          - Row 4 (Box 4 + Check 3): 4.3s - 4.9s → RIGHT
          - Row 5 (Box 5 + Check 4): 4.4s - 5.0s → RIGHT
          - Row 6 (Box 6 + Check 5): 4.5s - 5.1s → RIGHT
          - Row 7 (Box 7 + X 2):     4.6s - 5.2s → LEFT
          - Row 8 (Box 8 + X 3):     4.7s - 5.3s → LEFT
      
      KEY TIMING DECISIONS:
      ────────────────────────────────────────────────────────────────────────────────
      1. All 8 boxes remain in their original positions throughout
      2. Icons appear after boxes are fully visible (0.25s buffer)
      3. Icon appearances are staggered for visual interest (0.2s intervals)
      4. Fly-out starts after 0.9s hold with top-to-bottom stagger (0.1s)
      5. Total animation duration: 6s
      
      MODIFICATION GUIDELINES:
      ────────────────────────────────────────────────────────────────────────────────
      • To shift icon timing: Adjust icon begin times while maintaining stagger
      • To change fly-out timing: Adjust both box x-animate and icon translate together
      • To adjust stagger interval: Modify 0.1s/0.2s increments between animations
      • Fly-out directions: Checkmarks go RIGHT (600, 511), X icons go LEFT (-400, -511)
      • Phase boundaries: 0s → 1.7s → 4.0s → 5.3s → 6.0s
      -->

        <!-- ALL BOXES -->
        <!-- Box 1: y=24 (gets checkmark) -->
        <rect
          id="box1"
          x="89"
          y="24"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 46.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.0167; 0.0383; 0.06; 0.667; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.0167; 0.033; 0.667; 0.768; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #33B79D; #33B79D; #33B79D"
            keyTimes="0; 0.2833; 0.292; 0.667; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; 600; 89"
            keyTimes="0; 0.667; 0.767; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Box 2: y=82 (gets checkmark) -->
        <rect
          id="box2"
          x="89"
          y="82"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 104.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.0417; 0.0633; 0.085; 0.683; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.0417; 0.058; 0.683; 0.784; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #33B79D; #33B79D; #33B79D"
            keyTimes="0; 0.3167; 0.325; 0.683; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; 600; 89"
            keyTimes="0; 0.683; 0.783; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Box 3: y=140 (gets X icon) -->
        <rect
          id="box3"
          x="89"
          y="140"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 162.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.0667; 0.0883; 0.11; 0.7; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.0667; 0.083; 0.7; 0.8; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #E74D31; #E74D31; #E74D31"
            keyTimes="0; 0.35; 0.358; 0.7; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; -400; 89"
            keyTimes="0; 0.7; 0.8; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Box 4: y=198 (gets checkmark) -->
        <rect
          id="box4"
          x="89"
          y="198"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 220.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.0917; 0.1133; 0.135; 0.717; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.0917; 0.108; 0.717; 0.817; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #33B79D; #33B79D; #33B79D"
            keyTimes="0; 0.3833; 0.392; 0.717; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; 600; 89"
            keyTimes="0; 0.717; 0.817; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Box 5: y=256 (gets checkmark) -->
        <rect
          id="box5"
          x="89"
          y="256"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 278.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.1167; 0.1383; 0.16; 0.733; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.1167; 0.133; 0.733; 0.833; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #33B79D; #33B79D; #33B79D"
            keyTimes="0; 0.4167; 0.425; 0.733; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; 600; 89"
            keyTimes="0; 0.733; 0.833; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Box 6: y=314 (gets checkmark) -->
        <rect
          id="box6"
          x="89"
          y="314"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 336.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.1417; 0.1633; 0.185; 0.75; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.1417; 0.158; 0.75; 0.85; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #33B79D; #33B79D; #33B79D"
            keyTimes="0; 0.45; 0.458; 0.75; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; 600; 89"
            keyTimes="0; 0.75; 0.85; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Box 7: y=372 (gets X icon) -->
        <rect
          id="box7"
          x="89"
          y="372"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 394.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.1667; 0.1883; 0.21; 0.767; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.1667; 0.183; 0.767; 0.867; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #E74D31; #E74D31; #E74D31"
            keyTimes="0; 0.4833; 0.492; 0.767; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; -400; 89"
            keyTimes="0; 0.767; 0.867; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Box 8: y=430 (gets X icon) -->
        <rect
          id="box8"
          x="89"
          y="430"
          width="322"
          height="45"
          rx="10"
          stroke="#415CF5"
          stroke-width="4"
          opacity="0"
          transform-origin="250 452.5"
        >
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.1917; 0.2133; 0.235; 0.783; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.1917; 0.208; 0.783; 0.883; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke"
            values="#415CF5; #415CF5; #E74D31; #E74D31; #E74D31"
            keyTimes="0; 0.5167; 0.525; 0.783; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="x"
            values="89; 89; -400; 89"
            keyTimes="0; 0.783; 0.883; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
        </rect>

        <!-- Checkmark 1 (for box 1 at y=24) -->
        <g id="check1" opacity="0" transform-origin="379 46.8">
          <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M370.515 45.766C369.254 44.478 367.208 44.478 365.946 45.766C364.685 47.055 364.685 49.145 365.946 50.434L372.408 57.034C373.67 58.322 375.715 58.322 376.977 57.034L392.054 41.634C393.315 40.3447 393.315 38.2553 392.054 36.9666C390.792 35.6778 388.746 35.6778 387.485 36.9666L374.692 50.033L370.515 45.766Z"
            fill="#33B79D"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.2833; 0.3; 0.317; 0.667; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.2833; 0.292; 0.667; 0.767; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; 511,0; 0,0"
            keyTimes="0; 0.667; 0.767; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>

        <!-- Checkmark 2 (for box 2 at y=82) -->
        <g id="check2" opacity="0" transform-origin="379 104.8">
          <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M370.515 103.766C369.254 102.478 367.208 102.478 365.946 103.766C364.685 105.055 364.685 107.145 365.946 108.434L372.408 115.034C373.67 116.322 375.715 116.322 376.977 115.034L392.054 99.634C393.315 98.3447 393.315 96.2553 392.054 94.9666C390.792 93.6778 388.746 93.6778 387.485 94.9666L374.692 108.033L370.515 103.766Z"
            fill="#33B79D"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.3167; 0.333; 0.35; 0.683; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.3167; 0.325; 0.683; 0.783; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; 511,0; 0,0"
            keyTimes="0; 0.683; 0.783; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>

        <!-- X 1 (for box 3 at y=140) -->
        <g id="x1" opacity="0" transform-origin="379 163">
          <path
            d="M382.503 163L389.263 169.722C390.246 170.7 390.246 172.167 389.263 173.144C388.894 173.633 388.279 174 387.665 174C387.05 174 386.436 173.756 385.944 173.267L379.061 166.422L372.302 173.144C371.318 174.122 369.721 174.122 368.86 173.144C368.369 172.778 368 172.167 368 171.433C368 170.7 368.246 170.211 368.737 169.722L375.497 163L368.737 156.278C368.369 155.789 368 155.178 368 154.444C368 153.833 368.246 153.222 368.737 152.733C369.229 152.244 369.844 152 370.458 152C371.073 152 371.687 152.244 372.179 152.733L378.939 159.456L385.698 152.733C386.682 151.756 388.279 151.756 389.14 152.733C390.123 153.711 390.123 155.3 389.14 156.156L382.503 163Z"
            fill="#E74D31"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.35; 0.367; 0.383; 0.7; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.35; 0.358; 0.7; 0.8; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; -511,0; 0,0"
            keyTimes="0; 0.7; 0.8; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>

        <!-- Checkmark 3 (for box 4 at y=198) -->
        <g id="check3" opacity="0" transform-origin="379 220.8">
          <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M370.515 219.766C369.254 218.478 367.208 218.478 365.946 219.766C364.685 221.055 364.685 223.145 365.946 224.434L372.408 231.034C373.67 232.322 375.715 232.322 376.977 231.034L392.054 215.634C393.315 214.345 393.315 212.255 392.054 210.967C390.792 209.678 388.746 209.678 387.485 210.967L374.692 224.033L370.515 219.766Z"
            fill="#33B79D"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.3833; 0.4; 0.417; 0.717; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.3833; 0.392; 0.717; 0.817; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; 511,0; 0,0"
            keyTimes="0; 0.717; 0.817; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>

        <!-- Checkmark 4 (for box 5 at y=256) -->
        <g id="check4" opacity="0" transform-origin="379 278.8">
          <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M370.515 277.766C369.254 276.478 367.208 276.478 365.946 277.766C364.685 279.055 364.685 281.145 365.946 282.434L372.408 289.034C373.67 290.322 375.715 290.322 376.977 289.034L392.054 273.634C393.315 272.345 393.315 270.255 392.054 268.967C390.792 267.678 388.746 267.678 387.485 268.967L374.692 282.033L370.515 277.766Z"
            fill="#33B79D"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.4167; 0.433; 0.45; 0.733; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.4167; 0.425; 0.733; 0.833; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; 511,0; 0,0"
            keyTimes="0; 0.733; 0.833; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>

        <!-- Checkmark 5 (for box 6 at y=314) -->
        <g id="check5" opacity="0" transform-origin="379 336.8">
          <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M370.515 335.766C369.254 334.478 367.208 334.478 365.946 335.766C364.685 337.055 364.685 339.145 365.946 340.434L372.408 347.034C373.67 348.322 375.715 348.322 376.977 347.034L392.054 331.634C393.315 330.345 393.315 328.255 392.054 326.967C390.792 325.678 388.746 325.678 387.485 326.967L374.692 340.033L370.515 335.766Z"
            fill="#33B79D"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.45; 0.467; 0.483; 0.75; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.45; 0.458; 0.75; 0.85; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; 511,0; 0,0"
            keyTimes="0; 0.75; 0.85; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>

        <!-- X 2 (for box 7 at y=372) -->
        <g id="x2" opacity="0" transform-origin="379 395">
          <path
            d="M382.503 395L389.263 401.722C390.246 402.7 390.246 404.167 389.263 405.144C388.894 405.633 388.279 406 387.665 406C387.05 406 386.436 405.756 385.944 405.267L379.061 398.422L372.302 405.144C371.318 406.122 369.721 406.122 368.86 405.144C368.369 404.778 368 404.167 368 403.433C368 402.7 368.246 402.211 368.737 401.722L375.497 395L368.737 388.278C368.369 387.789 368 387.178 368 386.444C368 385.833 368.246 385.222 368.737 384.733C369.229 384.244 369.844 384 370.458 384C371.073 384 371.687 384.244 372.179 384.733L378.939 391.456L385.698 384.733C386.682 383.756 388.279 383.756 389.14 384.733C390.123 385.711 390.123 387.3 389.14 388.156L382.503 395Z"
            fill="#E74D31"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.4833; 0.5; 0.517; 0.767; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.4833; 0.492; 0.767; 0.867; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; -511,0; 0,0"
            keyTimes="0; 0.767; 0.867; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>

        <!-- X 3 (for box 8 at y=430) -->
        <g id="x3" opacity="0" transform-origin="379 453">
          <path
            d="M382.503 453L389.263 459.722C390.246 460.7 390.246 462.167 389.263 463.144C388.894 463.633 388.279 464 387.665 464C387.05 464 386.436 463.756 385.944 463.267L379.061 456.422L372.302 463.144C371.318 464.122 369.721 464.122 368.86 463.144C368.369 462.778 368 462.167 368 461.433C368 460.7 368.246 460.211 368.737 459.722L375.497 453L368.737 446.278C368.369 445.789 368 445.178 368 444.444C368 443.833 368.246 443.222 368.737 442.733C369.229 442.244 369.844 442 370.458 442C371.073 442 371.687 442.244 372.179 442.733L378.939 449.456L385.698 442.733C386.682 441.756 388.279 441.756 389.14 442.733C390.123 443.711 390.123 445.3 389.14 446.156L382.503 453Z"
            fill="#E74D31"
          />
          <animateTransform
            attributeName="transform"
            type="scale"
            values="1; 1; 1.3; 1; 1; 1"
            keyTimes="0; 0.5167; 0.533; 0.55; 0.783; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0; 0; 1; 1; 0; 0"
            keyTimes="0; 0.5167; 0.525; 0.783; 0.883; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            values="0,0; 0,0; -511,0; 0,0"
            keyTimes="0; 0.783; 0.883; 1"
            begin="0s"
            dur="6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </g>
      </g>
    </svg>
  </div>
  <div class="font-medium text-lg text-center mt-2">{title}</div>
  <div class="font-light text-center text-gray-500 max-w-md mt-2 text-balance">
    {description}
  </div>
  {#if warning}
    <div class="mt-6">
      <Warning
        warning_message={warning}
        warning_color="warning"
        warning_icon="exclaim"
      />
    </div>
  {/if}
</div>
