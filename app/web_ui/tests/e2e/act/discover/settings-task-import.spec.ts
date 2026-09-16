import { test, expect } from "../../fixtures"

test.describe("Settings - task and import", () => {
  /* @act
  ## Goals
  The create task page loads and displays the page title, subtitle with target project
  name, and the task creation form fields (task name, prompt/instructions, thinking
  instructions, input schema, output schema).

  ## Fixtures
  - registeredUser
  - seededProjectWithTask

  ## Hints
  - Route is /settings/create_task/{project_id}
  - Page title heading is "New Task"
  - Subtitle includes "A 'task' is a single goal"
  - Form fields: #task_name, #task_instructions, #thinking_instructions
  - Part headings: "Part 1: Overview", "Part 2: Task Type", "Part 3: Input Schema", "Part 4: Output Schema"
  - Has "Try an example." link for example task

  ## Assertions
  - Heading "New Task" is visible.
  - "Part 1: Overview" text is visible.
  - "Part 2: Task Type" text is visible.
  - "Part 3: Input Schema" text is visible.
  - "Part 4: Output Schema" text is visible.
  - Task name input is empty.
  - "Create Task" submit button is visible.
  - "Try an example." link is visible.
  */
  test("create task page loads with form fields", async ({
    page,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    const { project } = seededProjectWithTask

    await page.goto(`/settings/create_task/${project.id}`)

    await expect(
      page.getByRole("heading", { name: "New Task", exact: true }),
    ).toBeVisible()

    await expect(page.getByText("Part 1: Overview")).toBeVisible()
    await expect(page.getByText("Part 2: Task Type")).toBeVisible()
    await expect(page.getByText("Part 3: Input Schema")).toBeVisible()
    await expect(page.getByText("Part 4: Output Schema")).toBeVisible()

    await expect(page.locator("#task_name")).toHaveValue("")
    await expect(
      page.getByRole("button", { name: "Create Task" }),
    ).toBeVisible()
    await expect(page.getByText("Try an example.")).toBeVisible()
  })

  /* @act
  ## Goals
  Clicking the "Try an example." link on the create task page populates the form
  with the Joke Generator example task data.

  ## Fixtures
  - registeredUser
  - seededProjectWithTask

  ## Hints
  - "Try an example." is a button styled as a link.
  - After clicking, task name should be "Joke Generator".
  - Prompt/instructions should contain "Generate a joke, given a theme."

  ## Assertions
  - Task name field has value "Joke Generator".
  - Task instructions field contains "Generate a joke".
  */
  test("create task page fills example task", async ({
    page,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    const { project } = seededProjectWithTask

    await page.goto(`/settings/create_task/${project.id}`)

    await expect(page.locator("#task_name")).toHaveValue("")

    await page.getByText("Try an example.").click()

    await expect(page.locator("#task_name")).toHaveValue("Joke Generator")
    await expect(page.locator("#task_instructions")).toContainText("")
    const instructionValue = await page
      .locator("#task_instructions")
      .inputValue()
    expect(instructionValue).toContain("Generate a joke")
  })

  /* @act
  ## Goals
  Filling in the create task form and submitting creates a new task via POST.
  After submission the task exists in the backend.

  ## Fixtures
  - registeredUser
  - seededProjectWithTask
  - apiRequest

  ## Hints
  - Fill #task_name and #task_instructions, then click "Create Task".
  - After creation, verify via GET /api/projects/:pid/tasks that the new task exists.

  ## Assertions
  - After submit, the new task appears in the project's task list via the API.
  */
  test("create task page submits new task", async ({
    page,
    apiRequest,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    const { project } = seededProjectWithTask

    await page.goto(`/settings/create_task/${project.id}`)

    const taskName = `E2E Created Task ${Date.now()}`
    const taskInstruction = "Test instruction for e2e created task."

    await page.locator("#task_name").fill(taskName)
    await page.locator("#task_instructions").fill(taskInstruction)

    await page.getByRole("button", { name: "Create Task" }).click()

    await expect
      .poll(async () => {
        const resp = await apiRequest.get(
          `/api/projects/${encodeURIComponent(project.id)}/tasks`,
        )
        if (!resp.ok()) return false
        const tasks = (await resp.json()) as Array<{
          name: string
          instruction: string
        }>
        return tasks.some(
          (t) => t.name === taskName && t.instruction === taskInstruction,
        )
      })
      .toBeTruthy()
  })

  /* @act
  ## Goals
  The clone task page loads an existing task and pre-populates the form with
  "Copy of <task name>" as the name, and the original task instruction.

  ## Fixtures
  - registeredUser
  - seededProjectWithTask

  ## Hints
  - Route is /settings/clone_task/{project_id}/{task_id}
  - Page heading is "Clone Task"
  - Subtitle says "Create a new task, using an existing task as a template"
  - Task name field should have value "Copy of <original task name>"
  - Task instructions should match the original task's instruction

  ## Assertions
  - Heading "Clone Task" is visible.
  - Task name field has value starting with "Copy of ".
  - Task instructions field matches the seeded task instruction.
  - "Create Task" submit button is visible (since clone creates a new task).
  */
  test("clone task page loads with prefilled data", async ({
    page,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    const { project, task } = seededProjectWithTask

    await page.goto(`/settings/clone_task/${project.id}/${task.id}`)

    await expect(
      page.getByRole("heading", { name: "Clone Task", exact: true }),
    ).toBeVisible()

    await expect(page.locator("#task_name")).toHaveValue(`Copy of ${task.name}`)
    await expect(page.locator("#task_instructions")).toHaveValue(
      task.instruction,
    )

    await expect(
      page.getByRole("button", { name: "Create Task" }),
    ).toBeVisible()
  })

  /* @act
  ## Goals
  Submitting the clone task form creates a new task via POST. The cloned task
  appears in the project's task list with the "Copy of" prefix name.

  ## Fixtures
  - registeredUser
  - seededProjectWithTask
  - apiRequest

  ## Hints
  - Navigate to /settings/clone_task/{project_id}/{task_id}
  - Wait for the form to load, then click "Create Task"
  - Verify via GET /api/projects/:pid/tasks that a task named "Copy of <original>" exists

  ## Assertions
  - After submit, a task named "Copy of <original name>" exists in the API task list.
  */
  test("clone task page creates new task from clone", async ({
    page,
    apiRequest,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    const { project, task } = seededProjectWithTask

    await page.goto(`/settings/clone_task/${project.id}/${task.id}`)

    await expect(page.locator("#task_name")).toHaveValue(`Copy of ${task.name}`)

    await page.getByRole("button", { name: "Create Task" }).click()

    const expectedName = `Copy of ${task.name}`
    await expect
      .poll(async () => {
        const resp = await apiRequest.get(
          `/api/projects/${encodeURIComponent(project.id)}/tasks`,
        )
        if (!resp.ok()) return false
        const tasks = (await resp.json()) as Array<{ name: string }>
        return tasks.some((t) => t.name === expectedName)
      })
      .toBeTruthy()
  })

  /* @act
  ## Goals
  The import project page loads and shows the method selection screen with
  two options: "Import from Local Folder" and "Git Auto Sync".

  ## Fixtures
  - registeredUser
  - seededProjectWithTask

  ## Hints
  - Route is /settings/import_project
  - Page heading is "Import Project"
  - Two buttons: "Import from Local Folder" and "Git Auto Sync"
  - "Or create a new project" link at bottom, pointing to /settings/create_project

  ## Assertions
  - Heading "Import Project" is visible.
  - "Import from Local Folder" button text is visible.
  - "Git Auto Sync" button text is visible.
  - "create a new project" link is visible and points to /settings/create_project.
  */
  test("import project page shows method selection", async ({
    page,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    void seededProjectWithTask

    await page.goto("/settings/import_project")

    await expect(
      page.getByRole("heading", { name: "Import Project", exact: true }),
    ).toBeVisible()

    await expect(page.getByText("Import from Local Folder")).toBeVisible()
    await expect(page.getByText("Git Auto Sync")).toBeVisible()

    const createLink = page.getByRole("link", { name: "create a new project" })
    await expect(createLink).toBeVisible()
    await expect(createLink).toHaveAttribute("href", "/settings/create_project")
  })

  /* @act
  ## Goals
  Clicking "Import from Local Folder" on the import project page transitions
  to the local file import view showing a path input or file selector.

  ## Fixtures
  - registeredUser
  - seededProjectWithTask

  ## Hints
  - Click the "Import from Local Folder" button on the method selection screen.
  - The local file view shows "Select or enter the path to a project.kiln file"
  - If the file selector API is unavailable, it falls back to a manual path input
    with id="import_project_path".

  ## Assertions
  - After clicking "Import from Local Folder", the text "project.kiln file" is visible.
  */
  test("import project page local file shows path input", async ({
    page,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    void seededProjectWithTask

    await page.goto("/settings/import_project")

    await page.getByText("Import from Local Folder").click()

    await expect(
      page.getByText("Select or enter the path to a project.kiln file"),
    ).toBeVisible()
  })

  /* @act
  ## Goals
  The import project page displays breadcrumbs linking to Settings and Manage Projects.

  ## Fixtures
  - registeredUser
  - seededProjectWithTask

  ## Hints
  - Breadcrumbs are rendered by AppPage component.
  - First breadcrumb: "Settings" linking to /settings
  - Second breadcrumb: "Manage Projects" linking to /settings/manage_projects

  ## Assertions
  - "Settings" breadcrumb link is visible with href /settings.
  - "Manage Projects" breadcrumb link is visible with href /settings/manage_projects.
  */
  test("import project page shows breadcrumbs", async ({
    page,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    void seededProjectWithTask

    await page.goto("/settings/import_project")

    await expect(
      page.getByRole("heading", { name: "Import Project", exact: true }),
    ).toBeVisible()

    const settingsLink = page
      .locator(".breadcrumbs")
      .getByRole("link", { name: "Settings" })
    await expect(settingsLink).toBeVisible()
    await expect(settingsLink).toHaveAttribute("href", "/settings")

    const manageLink = page
      .locator(".breadcrumbs")
      .getByRole("link", { name: "Manage Projects" })
    await expect(manageLink).toBeVisible()
    await expect(manageLink).toHaveAttribute(
      "href",
      "/settings/manage_projects",
    )
  })

  /* @act
  ## Goals
  Clicking "Git Auto Sync" on the import project page transitions to the Git URL
  entry step with a progress bar and URL input.

  ## Fixtures
  - registeredUser
  - seededProjectWithTask

  ## Hints
  - Click the "Git Auto Sync" button on the method selection screen.
  - The URL step shows a progress bar and a form for entering a git repository URL.
  - The URL hash changes to #git.

  ## Assertions
  - After clicking "Git Auto Sync", the URL contains "#git".
  - A form for entering the git URL is visible.
  */
  test("import project page git option shows URL step", async ({
    page,
    registeredUser,
    seededProjectWithTask,
  }) => {
    void registeredUser
    void seededProjectWithTask

    await page.goto("/settings/import_project")

    await page.getByText("Git Auto Sync").click()

    await expect(page).toHaveURL(/.*#git$/)

    await expect(page.locator("#git_url")).toBeVisible()
  })
})
