<script lang="ts">
  export let generate_subtopics: () => void = () => {}
  export let generate_samples: () => void = () => {}
  export let project_id: string = ""
  export let task_id: string = ""
  export let is_setup: boolean = false
</script>

<div
  data-testid="data-gen-intro-stub"
  data-project-id={project_id}
  data-task-id={task_id}
  data-is-setup={is_setup}
  data-has-callbacks={!!generate_subtopics && !!generate_samples}
></div>
