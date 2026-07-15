// app.js – Handles landing, chat navigation, and backend communication

// Utility: fetch JSON with error handling
async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  return res.json();
}

// ---------- Landing page logic ----------
const nameForm = document.getElementById("name-form");
if (nameForm) {
  nameForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = document.getElementById("user-name").value.trim();
    if (name) {
      sessionStorage.setItem("userName", name);
      // Navigate to chat UI
      window.location.href = "chat.html";
    }
  });
}

// ---------- Chat page logic ----------
const msgForm = document.getElementById("msg-form");
if (msgForm) {
  let sessionId = null; // will be set after first successful request
  const messagesDiv = document.getElementById("messages");

  function addMessage(text, fromUser = true) {
    const el = document.createElement("div");
    el.textContent = text;
    el.style.margin = "0.5rem 0";
    el.style.textAlign = fromUser ? "right" : "left";
    el.style.color = fromUser ? "var(--accent)" : "var(--text)";
    messagesDiv.appendChild(el);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  msgForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("msg-input");
    const content = input.value.trim();
    if (!content) return;
    addMessage(content, true);
    input.value = "";
    try {
      const payload = { content };
      if (sessionId) payload.session_id = sessionId;
      const resp = await postJSON("http://localhost:8000/api/v1/chat/message", payload);
      sessionId = resp.session_id; // store for subsequent calls
      const reply = `Department: ${resp.department} (confidence: ${(resp.confidence * 100).toFixed(1)}%)`;
      addMessage(reply, false);
    } catch (err) {
      console.error(err);
      addMessage(`Error: ${err.message}`, false);
    }
  });
}

// ---------- Optional: show user name on chat page ----------
const brandNameSpan = document.querySelector(".brand-name");
if (brandNameSpan && sessionStorage.getItem("userName")) {
  brandNameSpan.textContent = `MediGuide – ${sessionStorage.getItem("userName")}`;
}
