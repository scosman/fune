import { test, expect } from "./fixtures"

/**
 * The eval creation flow:
 *   type picker (LLM Judges / Programmatic Checks) -> create form
 *
 * Every template's judge is implied (LLM judge, except tool call). The
 * programmatic checks are picked directly from the type page and create a
 * spec-less, template-less eval: the judge never reads a written rubric, so
 * no spec fields are asked for.
 */

test("programmatic check: type picker -> judge-only builder creates a template-less eval", async ({
  page,
  apiRequest,
  registeredUser,
  seededProjectWithTask,
}) => {
  void registeredUser
  const { project, task } = seededProjectWithTask

  // Seed a task run so the Test Judge pane has real data to test against.
  const run_resp = await apiRequest.post(
    `/api/projects/${project.id}/tasks/${task.id}/runs`,
    {
      data: {
        input: "Write a headline",
        output: "Calm report: no hate here",
        model_name: "gpt_4o",
        model_provider: "openai",
        adapter_name: "manual",
      },
    },
  )
  expect(run_resp.ok()).toBe(true)

  // Create Eval lands straight on the type picker with both sections
  await page.goto(`/specs/${project.id}/${task.id}`)
  await page.getByRole("button", { name: "Create Eval" }).first().click()
  await expect(page).toHaveURL(/select_template/)
  await expect(page.getByText("LLM Judges", { exact: true })).toBeVisible()
  await expect(
    page.getByText("Programmatic Checks", { exact: true }),
  ).toBeVisible()

  // A programmatic check: name + the judge's own config, and nothing else
  await page.getByText("Pattern Match").first().click()
  await expect(page).toHaveURL(/spec_builder\?.*judge=pattern_match/)
  await expect(
    page.getByText("Judge Configuration", { exact: true }),
  ).toBeVisible()
  await expect(page.getByLabel("Eval Name")).toBeVisible()
  await expect(page.getByRole("button", { name: "Save Eval" })).toBeVisible()

  // No spec is created for this flow, so no spec fields are asked for, and
  // no Kiln Pro creation path is offered anywhere in the builder.
  await expect(page.getByText("Issue Description")).toHaveCount(0)
  await expect(page.getByRole("button", { name: /Kiln Pro/ })).toHaveCount(0)

  await page.getByLabel("Eval Name").fill("No Hate Regex")
  await page.locator("#pattern_match_pattern").fill("\\bhate\\b")

  // Test the judge against the seeded run before saving: the draft endpoint
  // runs the not-yet-saved config against a transient eval.
  await expect(
    page.getByText("Test your judge on real data before saving."),
  ).toBeVisible()
  await page.getByTestId("run-test-btn").click()
  await expect(page.getByTestId("scores-section")).toBeVisible({
    timeout: 15000,
  })

  await page.getByRole("button", { name: "Save Eval" }).click()

  // Lands on the eval's detail page under the spec-less ("legacy") route
  await expect(page).toHaveURL(
    new RegExp(`/specs/${project.id}/${task.id}/legacy/[^/?]+$`),
    { timeout: 15000 },
  )

  // No spec was created
  const specs = await (
    await apiRequest.get(`/api/projects/${project.id}/tasks/${task.id}/specs`)
  ).json()
  expect(specs.length).toBe(0)

  // The eval is template-less (the user never claimed a template) but still
  // carries generated filters and priority/status. The list endpoint wraps the
  // evals so it can also report how many eval files this build couldn't read.
  const evals_response = await (
    await apiRequest.get(`/api/projects/${project.id}/tasks/${task.id}/evals`)
  ).json()
  expect(evals_response.load_error_count).toBe(0)
  expect(evals_response.evals.length).toBe(1)
  const evaluator = evals_response.evals[0]
  expect(evaluator.template).toBeNull()
  expect(evaluator.priority).toBe(1)
  expect(evaluator.status).toBe("active")
  // Splits are the only place filters live; the flat eval_set_filter_id /
  // train_set_filter_id fields are a load-time migration input and always null.
  expect(evaluator.splits.test.filter_id).toBe("tag::test_no_hate_regex")
  expect(evaluator.splits.train.filter_id).toBe("tag::train_no_hate_regex")

  // The judge was created with the eval and set as its default, so the eval
  // is ready to run rather than "Not Ready - Configure".
  const configs = await (
    await apiRequest.get(
      `/api/projects/${project.id}/tasks/${task.id}/evals/${evaluator.id}/eval_configs`,
    )
  ).json()
  expect(configs.length).toBe(1)
  expect(configs[0].properties.type).toBe("pattern_match")
  expect(configs[0].properties.pattern).toBe("\\bhate\\b")
  expect(evaluator.current_config_id).toBe(configs[0].id)

  // The default judge types endpoint reports the judge for the list page
  const judge_types = await (
    await apiRequest.get(
      `/api/projects/${project.id}/tasks/${task.id}/eval_default_judge_types`,
    )
  ).json()
  expect(judge_types[evaluator.id]).toBe("pattern_match")

  // The list page shows the eval with an editable priority/status and the
  // judge's label in the Type column (never "None" or "Legacy").
  await page.goto(`/specs/${project.id}/${task.id}`)
  const row = page.getByRole("row", { name: /No Hate Regex/ })
  await expect(
    row.getByText("Pattern Match", { exact: true }).first(),
  ).toBeVisible()
  await expect(row.getByLabel("Priority")).toHaveText(/P1/)
  await expect(row.getByLabel("Status")).toHaveText(/Active/)
  await expect(page.getByText("Legacy")).toHaveCount(0)
})

test("rubric templates go straight from the type picker to the spec form", async ({
  page,
  registeredUser,
  seededProjectWithTask,
}) => {
  void registeredUser
  const { project, task } = seededProjectWithTask

  await page.goto(`/specs/${project.id}/${task.id}/select_template`)
  await page.getByText("Toxicity", { exact: true }).first().click()

  // The template choice is the last question: no workflow screen in between.
  await expect(page).toHaveURL(/spec_builder\?.*judge=llm_judge/)
  await expect(page).toHaveURL(/type=toxicity/)
  await expect(page).toHaveURL(/workflow=manual/)
  await expect(
    page.getByText("Toxicity Examples", { exact: true }),
  ).toBeVisible()
  await expect(page.getByRole("button", { name: /Kiln Pro/ })).toHaveCount(0)
})

test("tool call check skips the tool dialog", async ({
  page,
  registeredUser,
  seededProjectWithTask,
}) => {
  void registeredUser
  const { project, task } = seededProjectWithTask

  await page.goto(`/specs/${project.id}/${task.id}/select_template`)
  await page.getByText("Tool Call Check", { exact: true }).first().click()

  // Straight to the builder with the tool call judge prefilled: the judge
  // carries its own expected-tool list, so the builder never has to ask which
  // tool the eval is about.
  await expect(page).toHaveURL(/spec_builder\?.*judge=tool_call_check/)
  await expect(page.getByText("Tool for this Eval")).toHaveCount(0)
  await expect(
    page.getByText("Judge Configuration", { exact: true }),
  ).toBeVisible()
  await expect(page.getByText("Expected Tools", { exact: true })).toBeVisible()
  await expect(page.getByRole("button", { name: "Save Eval" })).toBeVisible()
})

test("the Kiln Pro eval creation path is gone", async ({
  page,
  registeredUser,
  seededProjectWithTask,
}) => {
  void registeredUser
  const { project, task } = seededProjectWithTask

  // The workflow screen that offered Pro-vs-Manual no longer exists. Its old
  // URL now falls through to the eval detail route, which reports the missing
  // eval instead of offering a creation workflow.
  await page.goto(
    `/specs/${project.id}/${task.id}/select_workflow?type=toxicity&judge=llm_judge`,
  )
  await expect(page.getByText(/Spec not found/)).toBeVisible()
  await expect(
    page.getByText("Choose your Eval Creation Workflow"),
  ).toHaveCount(0)
  await expect(page.getByRole("button", { name: /Kiln Pro/ })).toHaveCount(0)

  // The builder has one mode. A hand-edited workflow param — the old pro
  // value, or garbage — still lands on the plain manual form.
  for (const workflow of ["pro", "GARBAGE"]) {
    await page.goto(
      `/specs/${project.id}/${task.id}/spec_builder?type=toxicity&workflow=${workflow}`,
    )
    await expect(
      page.getByRole("button", { name: "Create Eval" }),
    ).toBeVisible()
    await expect(page.getByRole("button", { name: /Kiln Pro/ })).toHaveCount(0)
  }
})

/**
 * The create form arrives pre-filled (autofilled name, judge defaults), so the
 * unsaved-changes guard has to key off real user edits -- otherwise just
 * opening the page and going back warns.
 */
function trackDialogs(page: import("@playwright/test").Page): string[] {
  const dialogs: string[] = []
  page.on("dialog", async (d) => {
    dialogs.push(d.message())
    await d.accept()
  })
  return dialogs
}

test("leaving an untouched create form does not warn about unsaved changes", async ({
  page,
  registeredUser,
  seededProjectWithTask,
}) => {
  void registeredUser
  const { project, task } = seededProjectWithTask
  const dialogs = trackDialogs(page)

  await page.goto(`/specs/${project.id}/${task.id}/select_template`)
  await page.getByText("Pattern Match").first().click()
  await expect(page.getByRole("button", { name: "Save Eval" })).toBeVisible()

  await page.goBack()
  await page.waitForTimeout(750)
  await expect(page).toHaveURL(/select_template/)
  expect(dialogs, "must not warn when nothing was touched").toEqual([])
})

test("leaving after editing the judge warns about unsaved changes", async ({
  page,
  registeredUser,
  seededProjectWithTask,
}) => {
  void registeredUser
  const { project, task } = seededProjectWithTask
  const dialogs = trackDialogs(page)

  // Navigate in-app so goBack is an SPA navigation: the guard's confirm()
  // carries a message there, unlike the browser's native beforeunload.
  await page.goto(`/specs/${project.id}/${task.id}/select_template`)
  await page.getByText("Pattern Match").first().click()
  await expect(page.getByRole("button", { name: "Save Eval" })).toBeVisible()
  await page.locator("#pattern_match_pattern").fill("^ok$")

  await page.goBack()
  await page.waitForTimeout(750)
  expect(dialogs.length, "must warn after editing the judge").toBe(1)
  expect(dialogs[0]).toContain("unsaved changes")
})

test("code judge: untouched does not warn, edited code does", async ({
  page,
  registeredUser,
  seededProjectWithTask,
}) => {
  void registeredUser
  const { project, task } = seededProjectWithTask
  const dialogs = trackDialogs(page)

  // The code editor doesn't emit bubbling input events, so it's tracked by value.
  await page.goto(
    `/specs/${project.id}/${task.id}/spec_builder?judge=code_eval&workflow=manual`,
  )
  await expect(page.getByRole("button", { name: "Save Eval" })).toBeVisible()
  await page.goBack()
  await page.waitForTimeout(750)
  expect(dialogs, "must not warn when the code is untouched").toEqual([])

  await page.goForward()
  await expect(page.getByRole("button", { name: "Save Eval" })).toBeVisible()
  await page.locator(".cm-content").click()
  await page.keyboard.type("# scoring")
  await page.goBack()
  await page.waitForTimeout(750)
  expect(dialogs.length, "must warn after a code edit").toBe(1)
})
