import { get } from "svelte/store"
import { indexedDBStore } from "./index_db_store"
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import type { Mock } from "vitest"

// Types for IndexedDB mocks
interface MockIDBRequest {
  result: unknown
  error: Error | null
  onsuccess: (() => void) | null
  onerror: ((event: Error) => void) | null
}

interface MockIDBOpenRequest extends MockIDBRequest {
  onupgradeneeded: (() => void) | null
  onblocked: (() => void) | null
}

interface MockIDBTransaction {
  objectStore: Mock<[string], MockIDBObjectStore>
  oncomplete: (() => void) | null
  onerror: ((event: Error) => void) | null
  onabort: (() => void) | null
  error: Error | null
}

interface MockIDBObjectStore {
  get: Mock<[string], MockIDBRequest>
  put: Mock<[{ key: string; value: unknown }], void>
}

interface MockIDBDatabase {
  transaction: Mock<[string[], string], MockIDBTransaction>
  objectStoreNames: {
    contains: Mock<[string], boolean>
  }
  createObjectStore: Mock<[string, { keyPath: string }], MockIDBObjectStore>
  close: Mock<[], void>
  onversionchange: (() => void) | null
}

// Mock IndexedDB for testing
let mockObjectStore: MockIDBObjectStore
let mockTransaction: MockIDBTransaction
let mockDatabase: MockIDBDatabase
let mockRequest: MockIDBOpenRequest
let mockIndexedDB: { open: Mock<[string, number], MockIDBOpenRequest> }

const DB_OPEN_TIMEOUT_MS = 5000

const newOpenRequest = (): MockIDBOpenRequest => ({
  result: mockDatabase,
  error: null,
  onsuccess: null,
  onerror: null,
  onupgradeneeded: null,
  onblocked: null,
})

// An open() that returns but never fires success, error or blocked.
const silentOpenRequest = (): MockIDBOpenRequest => {
  const request = newOpenRequest()
  mockIndexedDB.open.mockReturnValue(request)
  return request
}

const openRequestsFire = (event: "onsuccess" | "onblocked") => {
  mockIndexedDB.open.mockImplementation(() => {
    const request = newOpenRequest()
    process.nextTick(() => {
      const handler = request[event]
      if (handler) handler()
    })
    return request
  })
}

const autoResolveGet = (storedValue?: unknown) => {
  mockObjectStore.get.mockImplementation(() => {
    const request: MockIDBRequest = {
      result:
        storedValue === undefined
          ? null
          : { key: "test-key", value: storedValue },
      error: null,
      onsuccess: null,
      onerror: null,
    }
    process.nextTick(() => {
      if (request.onsuccess) request.onsuccess()
    })
    return request
  })
}

const transactionsFire = (event: "oncomplete" | "onabort") => {
  mockDatabase.transaction.mockImplementation(() => {
    const transaction: MockIDBTransaction = {
      objectStore: vi.fn<[string], MockIDBObjectStore>(() => mockObjectStore),
      oncomplete: null,
      onerror: null,
      onabort: null,
      error: null,
    }
    process.nextTick(() => {
      const handler = transaction[event]
      if (handler) handler()
    })
    return transaction
  })
}

const loggedOpenError = (consoleSpy: {
  mock: { calls: unknown[][] }
}): Error | undefined => {
  const call = consoleSpy.mock.calls.find(
    (args) => args[0] === "Failed to open IndexedDB:",
  )
  return call?.[1] as Error | undefined
}

describe("indexedDBStore", () => {
  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks()
    // The store arms a timeout on every open(), so drive time from the tests.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })

    // Create fresh mock objects
    mockObjectStore = {
      get: vi.fn<[string], MockIDBRequest>(),
      put: vi.fn<[{ key: string; value: unknown }], void>(),
    }

    mockTransaction = {
      objectStore: vi.fn<[string], MockIDBObjectStore>(() => mockObjectStore),
      oncomplete: null,
      onerror: null,
      onabort: null,
      error: null,
    }

    mockDatabase = {
      transaction: vi.fn<[string[], string], MockIDBTransaction>(
        () => mockTransaction,
      ),
      objectStoreNames: {
        contains: vi.fn<[string], boolean>(() => false),
      },
      createObjectStore: vi.fn<
        [string, { keyPath: string }],
        MockIDBObjectStore
      >(() => mockObjectStore),
      close: vi.fn<[], void>(),
      onversionchange: null,
    }

    mockRequest = {
      result: mockDatabase,
      error: null,
      onsuccess: null,
      onerror: null,
      onupgradeneeded: null,
      onblocked: null,
    }

    mockIndexedDB = {
      open: vi.fn<[string, number], MockIDBOpenRequest>(() => mockRequest),
    }

    // Reset mock behavior
    mockObjectStore.get.mockImplementation(() => ({
      result: null,
      error: null,
      onsuccess: null,
      onerror: null,
    }))

    mockObjectStore.put.mockImplementation(() => ({}))
    mockTransaction.objectStore.mockReturnValue(mockObjectStore)
    mockDatabase.transaction.mockReturnValue(mockTransaction)
    mockDatabase.objectStoreNames.contains.mockReturnValue(false)
    mockIndexedDB.open.mockReturnValue(mockRequest)

    // Mock browser environment
    vi.stubGlobal("window", {
      indexedDB: mockIndexedDB,
      localStorage: {
        getItem: vi.fn(),
        setItem: vi.fn(),
      },
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  describe("in browser environment", () => {
    it("should create a store with initial value", () => {
      const { store: storeInstance } = indexedDBStore(
        "test-key",
        "initial-value",
      )
      expect(get(storeInstance)).toBe("initial-value")
    })

    it("should initialize IndexedDB on first access", async () => {
      indexedDBStore("test-key", "initial-value")

      // Wait a bit for async initialization
      await vi.advanceTimersByTimeAsync(0)

      expect(mockIndexedDB.open).toHaveBeenCalledWith("kiln_stores", 1)
    })

    it("should create object store on upgrade", () => {
      indexedDBStore("test-key", "initial-value")

      // Simulate onupgradeneeded event
      if (mockRequest.onupgradeneeded) {
        mockRequest.onupgradeneeded()
      }

      expect(mockDatabase.createObjectStore).toHaveBeenCalledWith(
        "key_value_store",
        { keyPath: "key" },
      )
    })

    it("should load stored value from IndexedDB", async () => {
      const storedValue = "stored-value"

      // Mock the entire flow step by step
      mockObjectStore.get.mockImplementation(() => {
        const request: MockIDBRequest = {
          result: { key: "test-key", value: storedValue },
          error: null,
          onsuccess: null,
          onerror: null,
        }
        // Simulate immediate async success
        process.nextTick(() => {
          if (request.onsuccess) request.onsuccess()
        })
        return request
      })

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      // Trigger the DB initialization success which will trigger getValue
      process.nextTick(() => {
        if (mockRequest.onsuccess) {
          mockRequest.onsuccess()
        }
      })

      // Wait for all async operations to complete
      await initialized

      expect(get(storeInstance)).toBe(storedValue)
    })

    it("should handle IndexedDB initialization errors gracefully", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      // Simulate DB initialization error
      if (mockRequest.onerror) {
        mockRequest.error = new Error("DB failed to open")
        mockRequest.onerror(mockRequest.error)
      }

      // Wait for error handling
      await initialized

      // Store should still work with initial value
      expect(get(storeInstance)).toBe("initial-value")
      expect(consoleSpy).toHaveBeenCalledWith(
        "Failed to open IndexedDB:",
        expect.any(Error),
      )

      consoleSpy.mockRestore()
    })

    it("should handle get operation errors gracefully", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})

      // Mock get to throw an error
      mockObjectStore.get.mockImplementation(() => {
        const request: MockIDBRequest = {
          result: null,
          error: new Error("Get failed"),
          onsuccess: null,
          onerror: null,
        }
        process.nextTick(() => {
          if (request.onerror) request.onerror(request.error!)
        })
        return request
      })

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      // Trigger DB initialization success which will then allow getValue to proceed
      process.nextTick(() => {
        if (mockRequest.onsuccess) {
          mockRequest.onsuccess()
        }
      })

      await initialized

      // Should keep initial value on error
      expect(get(storeInstance)).toBe("initial-value")
      expect(consoleSpy).toHaveBeenCalledWith(
        "Failed to load initial value from IndexedDB:",
        expect.any(Error),
      )

      consoleSpy.mockRestore()
    })

    it("should handle null stored values correctly", async () => {
      // Mock get operation returning null
      const mockGetRequest: MockIDBRequest = {
        result: null,
        error: null,
        onsuccess: null,
        onerror: null,
      }

      mockObjectStore.get.mockImplementation(() => {
        process.nextTick(() => {
          if (mockGetRequest.onsuccess) mockGetRequest.onsuccess()
        })
        return mockGetRequest
      })

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      // Simulate successful DB initialization, which lets the get proceed
      process.nextTick(() => {
        if (mockRequest.onsuccess) {
          mockRequest.onsuccess()
        }
      })

      await initialized

      // Should keep initial value when no stored value found
      expect(get(storeInstance)).toBe("initial-value")
    })

    it("should create separate database connections for each store", async () => {
      // Clear any previous calls
      vi.clearAllMocks()

      // First store creation
      indexedDBStore("key1", "value1")
      await vi.advanceTimersByTimeAsync(10)

      // Second store creation
      indexedDBStore("key2", "value2")
      await vi.advanceTimersByTimeAsync(10)

      // Each store should create its own DB connection (db variable is scoped per function call)
      expect(mockIndexedDB.open).toHaveBeenCalledTimes(2)
    })
  })

  describe("in non-browser environment", () => {
    beforeEach(() => {
      // Remove window object to simulate non-browser environment
      vi.unstubAllGlobals()
    })

    it("should work without IndexedDB", () => {
      const { store: storeInstance } = indexedDBStore(
        "test-key",
        "initial-value",
      )
      expect(get(storeInstance)).toBe("initial-value")

      // Should not attempt to use IndexedDB
      expect(mockIndexedDB.open).not.toHaveBeenCalled()
    })

    it("should still be reactive without IndexedDB", () => {
      const { store: storeInstance } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      storeInstance.set("new-value")
      expect(get(storeInstance)).toBe("new-value")
    })
  })

  describe("persist", () => {
    it("should write current store value to IndexedDB and resolve when complete", async () => {
      // Setup: simulate successful DB open so setValue can work
      mockObjectStore.get.mockImplementation(() => {
        const request: MockIDBRequest = {
          result: null,
          error: null,
          onsuccess: null,
          onerror: null,
        }
        process.nextTick(() => {
          if (request.onsuccess) request.onsuccess()
        })
        return request
      })

      const {
        store: storeInstance,
        initialized,
        persist,
      } = indexedDBStore("test-key", "initial-value")

      // Trigger DB init success
      process.nextTick(() => {
        if (mockRequest.onsuccess) mockRequest.onsuccess()
      })
      await initialized

      // Update store value (this triggers an async write via subscriber)
      storeInstance.set("updated-value")

      // Mock the transaction for the persist call to complete
      mockDatabase.transaction.mockImplementation(() => {
        const tx: MockIDBTransaction = {
          objectStore: vi.fn<[string], MockIDBObjectStore>(
            () => mockObjectStore,
          ),
          oncomplete: null,
          onerror: null,
          onabort: null,
          error: null,
        }
        // Simulate transaction completing asynchronously
        process.nextTick(() => {
          if (tx.oncomplete) tx.oncomplete()
        })
        return tx
      })

      // persist() should resolve after IndexedDB write completes
      await persist()

      // Verify put was called with the updated value
      expect(mockObjectStore.put).toHaveBeenCalledWith({
        key: "test-key",
        value: "updated-value",
      })
    })

    it("should resolve immediately in non-browser environment", async () => {
      vi.unstubAllGlobals()
      const { persist } = indexedDBStore("test-key", "initial-value")
      // Should not throw or hang
      await persist()
    })
  })

  describe("edge cases", () => {
    it("should handle IndexedDB not being available", () => {
      // Mock browser environment without IndexedDB
      vi.stubGlobal("window", {
        localStorage: {
          getItem: vi.fn(),
          setItem: vi.fn(),
        },
      })

      const { store: storeInstance } = indexedDBStore(
        "test-key",
        "initial-value",
      )
      expect(get(storeInstance)).toBe("initial-value")

      // Should not crash when IndexedDB is not available
      storeInstance.set("new-value")
      expect(get(storeInstance)).toBe("new-value")
    })
  })

  describe("open() never leaves callers waiting", () => {
    it("rejects when the open is blocked, and initialized still resolves", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      openRequestsFire("onblocked")

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      await initialized

      expect(get(storeInstance)).toBe("initial-value")
      expect(loggedOpenError(consoleSpy)?.message).toContain("blocked")

      consoleSpy.mockRestore()
    })

    it("rejects with a real Error when the open fails without one", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      mockIndexedDB.open.mockImplementation(() => {
        const request = newOpenRequest()
        process.nextTick(() => {
          if (request.onerror) request.onerror(new Error("unused"))
        })
        return request
      })

      const { initialized } = indexedDBStore("test-key", "initial-value")
      await initialized

      // request.error is null here, so the store must supply its own Error.
      expect(loggedOpenError(consoleSpy)).toBeInstanceOf(Error)

      consoleSpy.mockRestore()
    })

    it("rejects via the timeout when the open fires no event at all", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      silentOpenRequest()

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      let settled = false
      initialized.then(() => {
        settled = true
      })

      await vi.advanceTimersByTimeAsync(DB_OPEN_TIMEOUT_MS - 1)
      expect(settled).toBe(false)

      await vi.advanceTimersByTimeAsync(1)
      await initialized

      expect(settled).toBe(true)
      expect(get(storeInstance)).toBe("initial-value")
      expect(loggedOpenError(consoleSpy)?.message).toContain("timed out")

      consoleSpy.mockRestore()
    })

    it("closes a connection that arrives after the timeout instead of caching it", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      const request = silentOpenRequest()

      const { initialized, persist } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      await vi.advanceTimersByTimeAsync(DB_OPEN_TIMEOUT_MS)
      await initialized

      if (request.onsuccess) request.onsuccess()

      expect(mockDatabase.close).toHaveBeenCalledTimes(1)
      expect(mockDatabase.transaction).not.toHaveBeenCalled()

      // The late connection was not cached, so a later write opens a new one.
      persist().catch(() => {})
      await vi.advanceTimersByTimeAsync(0)
      expect(mockIndexedDB.open).toHaveBeenCalledTimes(2)

      consoleSpy.mockRestore()
    })

    it("does not time out an open that succeeds in time", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      autoResolveGet()

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      process.nextTick(() => {
        if (mockRequest.onsuccess) mockRequest.onsuccess()
      })
      await initialized

      expect(vi.getTimerCount()).toBe(0)

      await vi.advanceTimersByTimeAsync(DB_OPEN_TIMEOUT_MS * 2)

      expect(get(storeInstance)).toBe("initial-value")
      expect(loggedOpenError(consoleSpy)).toBeUndefined()

      consoleSpy.mockRestore()
    })

    it("closes and forgets the connection when another tab requests an upgrade", async () => {
      autoResolveGet()

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      process.nextTick(() => {
        if (mockRequest.onsuccess) mockRequest.onsuccess()
      })
      await initialized

      expect(mockDatabase.onversionchange).toBeTypeOf("function")
      mockDatabase.onversionchange!()
      expect(mockDatabase.close).toHaveBeenCalledTimes(1)

      // The cached connection is cleared, so the next write re-opens the DB.
      storeInstance.set("new-value")
      await vi.advanceTimersByTimeAsync(0)
      expect(mockIndexedDB.open).toHaveBeenCalledTimes(2)
    })

    it("rejects persist() rather than hanging when the DB never opens", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      openRequestsFire("onblocked")

      const { initialized, persist } = indexedDBStore(
        "test-key",
        "initial-value",
      )

      await initialized
      await expect(persist()).rejects.toThrow(/blocked/)

      consoleSpy.mockRestore()
    })
  })

  describe("a failed load never overwrites stored state", () => {
    it("suppresses auto-save when the DB could not be opened", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      openRequestsFire("onblocked")

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )
      await initialized

      // A retry would now succeed: the stored row was never unreadable, this
      // tab just could not see it, so auto-save must not write defaults over it.
      openRequestsFire("onsuccess")
      transactionsFire("oncomplete")
      storeInstance.set("new-value")
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(0)

      expect(mockObjectStore.put).not.toHaveBeenCalled()

      consoleSpy.mockRestore()
    })

    it("suppresses auto-save when the read transaction aborts", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      openRequestsFire("onsuccess")
      transactionsFire("onabort")

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )
      await initialized

      transactionsFire("oncomplete")
      storeInstance.set("new-value")
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(0)

      expect(mockObjectStore.put).not.toHaveBeenCalled()

      consoleSpy.mockRestore()
    })

    it("still writes on an explicit persist() after a failed load", async () => {
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
      openRequestsFire("onblocked")

      const {
        store: storeInstance,
        initialized,
        persist,
      } = indexedDBStore("test-key", "initial-value")
      await initialized

      storeInstance.set("new-value")
      openRequestsFire("onsuccess")
      transactionsFire("oncomplete")
      await persist()

      expect(mockObjectStore.put).toHaveBeenCalledWith({
        key: "test-key",
        value: "new-value",
      })

      consoleSpy.mockRestore()
    })

    it("auto-saves as usual once the load has succeeded", async () => {
      openRequestsFire("onsuccess")
      autoResolveGet("stored-value")

      const { store: storeInstance, initialized } = indexedDBStore(
        "test-key",
        "initial-value",
      )
      await initialized
      expect(get(storeInstance)).toBe("stored-value")

      storeInstance.set("new-value")
      await vi.advanceTimersByTimeAsync(0)

      expect(mockObjectStore.put).toHaveBeenCalledWith({
        key: "test-key",
        value: "new-value",
      })
    })

    it("rejects persist() when the write transaction aborts", async () => {
      openRequestsFire("onsuccess")
      autoResolveGet()

      const { initialized, persist } = indexedDBStore(
        "test-key",
        "initial-value",
      )
      await initialized

      transactionsFire("onabort")
      await expect(persist()).rejects.toThrow(/aborted/)
    })
  })
})
