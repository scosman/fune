import { writable } from "svelte/store"

// An open() that never fires success, error or blocked would leave every
// caller awaiting forever, so bound it.
const DB_OPEN_TIMEOUT_MS = 5000

// A load that failed is not the same as a key that has never been written:
// only the second one means the store may safely mirror its value back.
type StoredValueLoad<T> = { loaded: true; value: T | null } | { loaded: false }

// Custom function to create an IndexedDB-backed store
export function indexedDBStore<T>(key: string, initialValue: T) {
  // Check if IndexedDB is available
  const isBrowser = typeof window !== "undefined" && window.indexedDB

  const store = writable(initialValue)
  const DB_NAME = "kiln_stores"
  const STORE_NAME = "key_value_store"
  const DB_VERSION = 1

  let initPromise: Promise<void>

  if (isBrowser) {
    let db: IDBDatabase | null = null
    let autoSaveEnabled = false

    // Initialize IndexedDB
    const initDB = (): Promise<IDBDatabase> => {
      return new Promise((resolve, reject) => {
        if (db) {
          resolve(db)
          return
        }

        const request = window.indexedDB.open(DB_NAME, DB_VERSION)

        let settled = false
        let openTimeout: ReturnType<typeof setTimeout> | undefined

        const claimSettle = (): boolean => {
          if (settled) {
            return false
          }
          settled = true
          if (openTimeout !== undefined) {
            clearTimeout(openTimeout)
            openTimeout = undefined
          }
          return true
        }

        request.onerror = () => {
          if (!claimSettle()) {
            return
          }
          const error =
            request.error ?? new Error(`Failed to open IndexedDB "${DB_NAME}"`)
          console.error("Failed to open IndexedDB:", error)
          reject(error)
        }

        request.onblocked = () => {
          if (!claimSettle()) {
            return
          }
          const error = new Error(
            `Opening IndexedDB "${DB_NAME}" was blocked by another open connection`,
          )
          console.error("Failed to open IndexedDB:", error)
          reject(error)
        }

        request.onsuccess = () => {
          const database = request.result
          if (!claimSettle()) {
            // This open already timed out, so close the late connection rather
            // than leave it blocking other tabs.
            database.close()
            return
          }
          database.onversionchange = () => {
            database.close()
            if (db === database) {
              db = null
            }
          }
          db = database
          resolve(database)
        }

        request.onupgradeneeded = () => {
          const database = request.result
          if (!database.objectStoreNames.contains(STORE_NAME)) {
            database.createObjectStore(STORE_NAME, { keyPath: "key" })
          }
        }

        openTimeout = setTimeout(() => {
          if (!claimSettle()) {
            return
          }
          const error = new Error(
            `Opening IndexedDB "${DB_NAME}" timed out after ${DB_OPEN_TIMEOUT_MS}ms`,
          )
          console.error("Failed to open IndexedDB:", error)
          reject(error)
        }, DB_OPEN_TIMEOUT_MS)
      })
    }

    // Get value from IndexedDB
    const getValue = async (): Promise<StoredValueLoad<T>> => {
      try {
        const database = await initDB()
        const transaction = database.transaction([STORE_NAME], "readonly")
        const objectStore = transaction.objectStore(STORE_NAME)
        const request = objectStore.get(key)

        return new Promise((resolve, reject) => {
          request.onsuccess = () => {
            const result = request.result
            resolve({ loaded: true, value: result ? result.value : null })
          }
          request.onerror = () =>
            reject(
              request.error ??
                new Error(`Failed to read IndexedDB key "${key}"`),
            )
          // An aborted transaction does not always surface as a request error.
          transaction.onabort = () =>
            reject(
              transaction.error ??
                new Error(`IndexedDB read of key "${key}" was aborted`),
            )
        })
      } catch (error) {
        console.error("Failed to get value from IndexedDB:", error)
        return { loaded: false }
      }
    }

    // Set value in IndexedDB
    const setValue = async (value: T): Promise<void> => {
      let database: IDBDatabase
      try {
        database = await initDB()
      } catch (error) {
        console.error(
          `Failed to initialize DB for setValue (key: ${key}):`,
          error,
        )
        throw error
      }

      try {
        const transaction = database.transaction([STORE_NAME], "readwrite")
        const objectStore = transaction.objectStore(STORE_NAME)
        objectStore.put({ key, value })

        return new Promise((resolve, reject) => {
          transaction.oncomplete = () => resolve()
          transaction.onerror = () => {
            reject(
              transaction.error ??
                new Error(`IndexedDB write of key "${key}" failed`),
            )
          }
          // An aborted transaction does not always fire an error event.
          transaction.onabort = () => {
            reject(
              transaction.error ??
                new Error(`IndexedDB write of key "${key}" was aborted`),
            )
          }
        })
      } catch (error) {
        console.error(
          `Error setting up transaction/put in setValue (key: ${key}):`,
          error,
        )
        throw error
      }
    }

    // Load initial value from IndexedDB
    initPromise = getValue()
      .then((load) => {
        if (!load.loaded) {
          return
        }
        try {
          if (load.value !== null) {
            store.set(load.value)
          }
        } finally {
          // A load that failed leaves this false: mirroring store writes back
          // would overwrite a stored value we were never able to read.
          autoSaveEnabled = true
        }
      })
      .catch((error) => {
        console.error("Failed to load initial value from IndexedDB:", error)
      })
      .finally(() => {
        if (!autoSaveEnabled) {
          console.warn(
            `IndexedDB auto-save is disabled for key "${key}": its stored value could not be read`,
          )
        }
      })

    // Subscribe to changes and update IndexedDB
    store.subscribe((value) => {
      if (autoSaveEnabled) {
        setValue(value).catch((error) => {
          console.error("Failed to update IndexedDB:", error)
        })
      }
    })
    // Flush the current store value to IndexedDB, resolving when the write completes
    const persist = async (): Promise<void> => {
      await initPromise
      let currentValue: T | undefined
      const unsub = store.subscribe((v) => (currentValue = v))
      unsub()
      return setValue(currentValue as T)
    }

    return {
      store,
      initialized: initPromise,
      persist,
    }
  } else {
    // If not in browser, resolve immediately
    initPromise = Promise.resolve()

    return {
      store,
      initialized: initPromise,
      persist: () => Promise.resolve(),
    }
  }
}
