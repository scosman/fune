<script lang="ts">
  // "Conversation generation" animation: four chat bubbles arrive one at a
  // time (alternating speakers), then the whole thread flies off to clear the
  // slate for the next generation. Used on the eval builder's conversation-
  // building screens (plan drafting, driving, review preparation) so those
  // waits read as "Kiln is building a conversation" rather than a bare spinner.
  // Same title/description/warning contract as the other themed animations.
  import Warning from "$lib/ui/warning.svelte"
  export let title: string
  export let description: string
  export let warning: string | null = null
</script>

<div class="flex flex-col items-center justify-center">
  <div class="flex flex-col max-h-[250px] max-w-[250px] mt-6">
    <svg
      width="250"
      height="232"
      viewBox="0 0 500 465"
      fill="none"
      overflow="hidden"
      xmlns="http://www.w3.org/2000/svg"
    >
      <!--
      ╔══════════════════════════════════════════════════════════════════════════════╗
      ║  CONVERSATION GENERATION — ANIMATION TIMELINE                                ║
      ║  Concept: an AI building a synthetic conversation. Four chat bubbles arrive   ║
      ║  one at a time (alternating speakers), then the whole thread flies off to     ║
      ║  the right, clearing the slate for the next generation.                       ║
      ╠══════════════════════════════════════════════════════════════════════════════╣
      ║  MASTER CYCLE: 5.0s, repeatCount="indefinite"                                 ║
      ║  Every <animate>/<animateTransform> below shares dur="5s" and encodes the      ║
      ║  FULL cycle via keyTimes. This is what keeps the loop perfectly in sync and    ║
      ║  seamless — do NOT convert these to begin="Ns" offsets with fill="freeze",     ║
      ║  that breaks looping.                                                         ║
      ║                                                                               ║
      ║  keyTime  =  seconds / 5.0     (0.1 = 0.5s,  0.2 = 1.0s,  0.8 = 4.0s ...)      ║
      ╠══════════════════════════════════════════════════════════════════════════════╣
      ║  PHASE 1 — ARRIVAL (0.0s → 3.5s)                                              ║
      ║  Each bubble: opacity 0 → 1  AND  translateY +15px → 0  over 500ms.            ║
      ║  Stagger: a new bubble starts every 1.0s (500ms animation + 500ms rest).       ║
      ║                                                                               ║
      ║    Bubble 1 (speaker A, right)  0.0s → 0.5s    keyTimes 0.0  → 0.1             ║
      ║    Bubble 2 (speaker B, left)   1.0s → 1.5s    keyTimes 0.2  → 0.3             ║
      ║    Bubble 3 (speaker A, right)  2.0s → 2.5s    keyTimes 0.4  → 0.5             ║
      ║    Bubble 4 (speaker B, left)   3.0s → 3.5s    keyTimes 0.6  → 0.7             ║
      ║                                                                               ║
      ║  Easing — opacity  : keySplines "0 0 0.2 1"      (ease-out, reads fast)        ║
      ║  Easing — translate: keySplines "0.16 1 0.3 1"   (expo-out, settles softly)    ║
      ╠══════════════════════════════════════════════════════════════════════════════╣
      ║  PHASE 2 — HOLD (3.5s → 4.0s)                                                 ║
      ║  500ms beat with all four bubbles at rest. Same 500ms rhythm as the stagger.   ║
      ╠══════════════════════════════════════════════════════════════════════════════╣
      ║  PHASE 3 — EXIT (4.0s → 5.0s, 1.0s total)                                     ║
      ║  The whole thread (#stage) translates right as one rigid group.                ║
      ║    4.00s → 4.30s   x: 0 → -10    anticipation / wind-up  (ease-in-out)         ║
      ║    4.30s → 5.00s   x: -10 → 620  fly off right           (ease-in, accelerate) ║
      ║  620px clears the leftmost bubble (x=30, so it needs >490) with margin.        ║
      ║  Bubbles keep opacity 1 — they leave, they don't fade.                         ║
      ║  At 5.0s everything snaps back to its t=0 state (offscreen + opacity 0), so    ║
      ║  the seam is invisible and the loop restarts with no dead frame.               ║
      ╚══════════════════════════════════════════════════════════════════════════════╝
      -->

      <!-- #stage: holds all four bubbles so PHASE 3 moves them as one rigid thread -->
      <g id="stage">
        <animateTransform
          attributeName="transform"
          type="translate"
          additive="replace"
          dur="5s"
          repeatCount="indefinite"
          calcMode="spline"
          keyTimes="0;0.8;0.86;1"
          values="0 0;0 0;-10 0;620 0"
          keySplines="0 0 1 1;0.33 0 0.67 1;0.5 0 0.85 0.35"
        />

        <!-- Bubble 1: speaker A (right, #CCCCCC) — x=146 y=29 — arrives 0.0s → 0.5s -->
        <g opacity="0">
          <animate
            attributeName="opacity"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.1;1"
            values="0;1;1"
            keySplines="0 0 0.2 1;0 0 1 1"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.1;1"
            values="0 15;0 0;0 0"
            keySplines="0.16 1 0.3 1;0 0 1 1"
          />
          <rect x="146" y="29" width="326" height="81" rx="12" fill="#CCCCCC" />
        </g>

        <!-- Bubble 2: speaker B (left, #F1F1F1) — x=30 y=135 — arrives 1.0s → 1.5s -->
        <g opacity="0">
          <animate
            attributeName="opacity"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.2;0.3;1"
            values="0;0;1;1"
            keySplines="0 0 1 1;0 0 0.2 1;0 0 1 1"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.2;0.3;1"
            values="0 15;0 15;0 0;0 0"
            keySplines="0 0 1 1;0.16 1 0.3 1;0 0 1 1"
          />
          <rect x="30" y="135" width="326" height="81" rx="12" fill="#F1F1F1" />
        </g>

        <!-- Bubble 3: speaker A (right, #CCCCCC) — x=146 y=241 — arrives 2.0s → 2.5s -->
        <g opacity="0">
          <animate
            attributeName="opacity"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.4;0.5;1"
            values="0;0;1;1"
            keySplines="0 0 1 1;0 0 0.2 1;0 0 1 1"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.4;0.5;1"
            values="0 15;0 15;0 0;0 0"
            keySplines="0 0 1 1;0.16 1 0.3 1;0 0 1 1"
          />
          <rect
            x="146"
            y="241"
            width="326"
            height="81"
            rx="12"
            fill="#CCCCCC"
          />
        </g>

        <!-- Bubble 4: speaker B (left, #F1F1F1) — x=30 y=347 — arrives 3.0s → 3.5s -->
        <g opacity="0">
          <animate
            attributeName="opacity"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.6;0.7;1"
            values="0;0;1;1"
            keySplines="0 0 1 1;0 0 0.2 1;0 0 1 1"
          />
          <animateTransform
            attributeName="transform"
            type="translate"
            dur="5s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.6;0.7;1"
            values="0 15;0 15;0 0;0 0"
            keySplines="0 0 1 1;0.16 1 0.3 1;0 0 1 1"
          />
          <rect x="30" y="347" width="326" height="81" rx="12" fill="#F1F1F1" />
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
