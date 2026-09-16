<script lang="ts">
  import { onDestroy } from "svelte"

  let dotCount = 1
  const dotInterval = setInterval(() => {
    dotCount = (dotCount % 3) + 1
  }, 2000)
  onDestroy(() => clearInterval(dotInterval))
</script>

<div class="flex items-center py-0.5">
  <img src="/images/chat_icon_animated.svg" alt="loading..." class="w-9 h-9" />
  <span class="text-sm ml-2 text-gray-500">Thinking{".".repeat(dotCount)}</span>
</div>
