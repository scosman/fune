<script lang="ts">
  import FormContainer from "$lib/utils/form_container.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import { type KilnError, createKilnError } from "$lib/utils/error_handlers"
  import Warning from "$lib/ui/warning.svelte"
  import { client } from "$lib/api_client"
  import { setup_ph_user } from "$lib/utils/connect_ph"
  import {
    redirect_to_personal,
    redirect_after_registration,
  } from "../registration_helpers"
  import { agentInfo } from "$lib/agent"

  agentInfo.set({
    name: "Setup: Work Registration",
    description:
      "Onboarding step to register a work account with name, email, and organization.",
  })

  let email = ""
  let full_name = ""
  let allow_personal_email_domains = false
  let entered_personal_email = false
  let loading = false
  let error: KilnError | null = null

  async function switch_to_personal() {
    loading = true
    error = null
    try {
      const { error: settings_error } = await client.POST("/api/settings", {
        body: {
          user_type: "personal",
          work_use_contact: null,
        },
      })
      if (settings_error) {
        throw settings_error
      }
      redirect_to_personal()
    } catch (e) {
      console.error("Error switching to personal", e)
      error = createKilnError(e)
    } finally {
      loading = false
    }
  }

  async function register() {
    loading = true
    error = null
    try {
      if (!allow_personal_email_domains && check_personal_email_domain(email)) {
        entered_personal_email = true
        return
      }
      if (!full_name) {
        throw new Error("Full name is required")
      }
      const { error } = await client.POST("/api/settings", {
        body: {
          user_type: "work",
          work_use_contact: email,
        },
      })
      if (error) {
        throw error
      }
      const res = await fetch("https://kiln.tech/api/subscribe_to_newsletter", {
        method: "POST",
        body: JSON.stringify({
          email,
          work_use: "true",
          full_name,
        }),
        headers: {
          "Content-Type": "application/json",
        },
      })
      if (res.status !== 200) {
        throw new Error("Failed to register")
      }

      // No need to await this
      setup_ph_user()

      redirect_after_registration()
    } catch (e) {
      error = createKilnError(e)
    } finally {
      loading = false
    }
  }

  function check_personal_email_domain(email: string): boolean {
    const lower_email = email.toLowerCase()
    return (
      lower_email.includes("@gmail.com") ||
      lower_email.includes("@yahoo.com") ||
      lower_email.includes("@hotmail.com") ||
      lower_email.includes("@outlook.com") ||
      lower_email.includes("@icloud.com") ||
      lower_email.includes("@aol.com") ||
      lower_email.includes("@verizon.net") ||
      lower_email.includes("@comcast.net") ||
      lower_email.includes("@sbcglobal.net") ||
      lower_email.includes("@att.net") ||
      lower_email.includes("@yahoo.co.uk") ||
      lower_email.includes("@yahoo.com.au") ||
      lower_email.includes("@qq.com") ||
      lower_email.includes("@163.com") ||
      lower_email.includes("@126.com") ||
      lower_email.includes("@yeah.net") ||
      lower_email.includes("@sina.com")
    )
  }
</script>

<div class="flex-none flex flex-row items-center justify-center">
  <img src="/logo.svg" alt="logo" class="size-8 mb-3" />
</div>
<h1 class="text-2xl lg:text-4xl flex-none font-bold text-center">
  Register for Work Use
</h1>
<h3
  class="text-base font-medium text-center mt-3 max-w-[600px] mx-auto text-balance"
>
  Kiln is currently free to use at work — just register with your work email to
  unlock commercial use.
</h3>

<div
  class="flex-none py-4 h-full flex flex-col w-full mx-auto items-center justify-center"
>
  <div class="max-w-[280px] mx-auto mt-2">
    <FormContainer
      on:submit={register}
      submit_label="Continue"
      keyboard_submit={false}
      submitting={loading}
      {error}
    >
      <FormElement
        id="full_name"
        inputType="input"
        label="Full Name"
        bind:value={full_name}
      />
      <FormElement
        id="email"
        inputType="input"
        label="Work Email"
        bind:value={email}
      />
      {#if entered_personal_email}
        <div class="flex flex-col gap-2">
          <FormElement
            id="allow_personal_email_domains"
            inputType="checkbox"
            label="This is my work email address"
            bind:value={allow_personal_email_domains}
          />
          <Warning
            warning_message="This looks like a personal email address. You must attest this is your primary work email address to unlock commercial usage."
          />
        </div>
      {/if}
    </FormContainer>
  </div>
</div>

<div class="text-center font-light mt-4">
  Or <button class="link" on:click={switch_to_personal}
    >switch to personal non-commercial use</button
  >.
</div>
