<script lang="ts">
  import { ui_state } from "$lib/stores"
  import CopilotAuthPage from "$lib/ui/kiln_copilot/copilot_auth_page.svelte"
  import { agentInfo } from "$lib/agent"
  import {
    data_guide_return,
    stashed_data_guide_caller,
    with_data_guide_caller,
  } from "$lib/utils/data_guide_return"

  agentInfo.set({
    name: "Data Guide Kiln Pro Auth",
    description:
      "Authentication page for Kiln Pro access to set up an input data guide.",
  })

  // Static route (no project/task in the path) so the Kinde OAuth redirect_uri
  // is a fixed URL. The destination is resolved from the current project/task.
  $: project_id = $ui_state.current_project_id
  $: task_id = $ui_state.current_task_id
  // The caller cannot ride this URL (OAuth returns to it verbatim), so the
  // page that sent the user here stashed it for the round trip.
  const caller = stashed_data_guide_caller()
  $: return_target = data_guide_return(caller, project_id ?? "", task_id ?? "")
</script>

<CopilotAuthPage
  title="Set Up Data Guide"
  subtitle="Your Data Guide will help us generate better synthetic inputs."
  docs_link="https://docs.kiln.tech/docs/synthetic-data-generation"
  breadcrumbs={[{ label: return_target.label, href: return_target.href }]}
  success_redirect_url={with_data_guide_caller(
    `/generate/${project_id}/${task_id}/data_guide_setup_copilot`,
    caller,
  )}
/>
