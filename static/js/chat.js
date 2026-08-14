// LocalStorage Persistence
let savedPages = JSON.parse(localStorage.getItem("savedPages")) || [];
let activePageIndex = null;

// Speech Recognition instance
let recognition = null;
let isListening = false;
let currentSpeechTarget = 'chat';

window.addEventListener("DOMContentLoaded", () => {
  renderSidebar();
  
  // Outside clicks hide contextual menus
  document.addEventListener("click", (e) => {
    const dropdown = document.getElementById("dropdown-menu");
    const btn = document.getElementById("dropdown-btn");
    if (dropdown && !dropdown.contains(e.target) && !btn.contains(e.target)) {
      dropdown.classList.remove("show");
    }

    const contextMenu = document.getElementById("context-menu");
    if (contextMenu) contextMenu.style.display = "none";
  });

  initSpeechRecognition();
});

function toggleDropdown() {
  const menu = document.getElementById("dropdown-menu");
  if (menu) menu.classList.toggle("show");
}

function switchWorkspace(viewType) {
  const chatView = document.getElementById("chat-workspace");
  const imageView = document.getElementById("image-workspace");

  if (viewType === 'image') {
    chatView.classList.add("hidden-workspace");
    imageView.classList.remove("hidden-workspace");
  } else {
    imageView.classList.add("hidden-workspace");
    chatView.classList.remove("hidden-workspace");
  }
}

function openImageGenPage(e) {
  e.preventDefault();
  const dropdown = document.getElementById("dropdown-menu");
  if (dropdown) dropdown.classList.remove("show");
  switchWorkspace('image');
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const chatArea = document.getElementById("chat-workspace");
  const imageArea = document.getElementById("image-workspace");
  const toggleBtn = document.getElementById("sidebar-toggle-btn");

  if (!sidebar || !chatArea || !toggleBtn) return;

  sidebar.classList.toggle("collapsed");
  chatArea.classList.toggle("expanded");
  if (imageArea) imageArea.classList.toggle("expanded");
  toggleBtn.classList.toggle("sidebar-collapsed");

  toggleBtn.innerHTML = sidebar.classList.contains("collapsed") ? "»" : "«";
}

// Clean text structure parser (removes asterisks & hash markdown)
function formatBotResponse(text) {
  if (!text) return "";

  let formatted = text.replace(/^#{1,6}\s*(.*)$/gm, '<h3 style="margin: 10px 0 6px 0; color: #030438; font-size: 16px;">$1</h3>');
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

  const lines = formatted.split("\n");
  let inList = false;
  let result = [];

  lines.forEach(line => {
    let trimmed = line.trim();
    if (/^[\-\*\+]\s+/.test(trimmed)) {
      let content = trimmed.replace(/^[\-\*\+]\s+/, '');
      if (!inList) {
        inList = true;
        result.push('<ul style="margin: 6px 0 6px 20px; padding-left: 10px;">');
      }
      result.push(`<li style="margin-bottom: 4px;">${content}</li>`);
    } else {
      if (inList) {
        inList = false;
        result.push('</ul>');
      }
      if (trimmed.length > 0 && !trimmed.startsWith('<h3')) {
        result.push(`<p style="margin-bottom: 8px;">${trimmed}</p>`);
      } else {
        result.push(trimmed);
      }
    }
  });

  if (inList) result.push('</ul>');
  return result.join("\n");
}

// Function to trigger image file download to PC
function downloadImage(imageUrl, fileName) {
  fetch(imageUrl)
    .then(response => response.blob())
    .then(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = fileName || 'generated-image.jpg';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    })
    .catch(() => alert('Failed to download image.'));
}

// Image Generator Handler
function generateImage() {
  const input = document.getElementById("image-prompt-input");
  const prompt = input.value.trim();
  if (!prompt) return;

  const gallery = document.getElementById("image-gallery-window");

  // Add User Prompt message
  const userMsg = document.createElement("div");
  userMsg.className = "user-msg";
  userMsg.textContent = "Image Prompt: " + prompt;
  gallery.appendChild(userMsg);

  // Save new session entry to Sidebar
  saveNewPageSession(prompt, 'image');

  // Add Card Container
  const botCard = document.createElement("div");
  botCard.className = "generated-image-card";
  botCard.innerHTML = `<p>🎨 <strong>Prompt:</strong> ${prompt}</p><p><em>Generating your image...</em></p>`;
  gallery.appendChild(botCard);
  gallery.scrollTop = gallery.scrollHeight;

  fetch("/generate_image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: prompt })
  })
    .then(res => res.json())
    .then(data => {
      if (data.image_url) {
        const imageName = prompt.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 20) + ".jpg";
        botCard.innerHTML = `
          <p>🎨 <strong>Prompt:</strong> ${prompt}</p>
          <img src="${data.image_url}" alt="${prompt}">
          <div class="image-actions">
            <button class="download-btn" onclick="downloadImage('${data.image_url}', '${imageName}')">
              📥 Download Image
            </button>
          </div>
        `;
      } else {
        botCard.innerHTML = `<p>🎨 <strong>Prompt:</strong> ${prompt}</p><p style="color: #ff5555;">Failed to generate image.</p>`;
      }
      gallery.scrollTop = gallery.scrollHeight;
      updateCurrentPageSession();
    })
    .catch(() => {
      botCard.innerHTML = `<p>🎨 <strong>Prompt:</strong> ${prompt}</p><p style="color: #ff5555;">Error contacting server.</p>`;
    });

  input.value = "";
}

// Web Speech API
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    document.querySelectorAll(".speech-btn").forEach(btn => btn.style.display = "none");
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    isListening = true;
    showSpeechPopup(true);
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    const targetInputId = currentSpeechTarget === 'image' ? "image-prompt-input" : "user-input";
    const input = document.getElementById(targetInputId);
    if (input) input.value = transcript;
  };

  recognition.onerror = () => stopSpeechRecognition();
  recognition.onend = () => stopSpeechRecognition();
}

function toggleSpeechRecognition(target = 'chat') {
  currentSpeechTarget = target;
  if (!recognition) {
    alert("Speech recognition is not supported in your browser.");
    return;
  }

  if (isListening) {
    stopSpeechRecognition();
  } else {
    try { recognition.start(); } catch (e) { console.error(e); }
  }
}

function stopSpeechRecognition() {
  if (recognition && isListening) recognition.stop();
  isListening = false;
  showSpeechPopup(false);
}

function showSpeechPopup(show) {
  const popup = document.getElementById("speech-popup");
  if (!popup) return;
  if (show) popup.classList.remove("hidden");
  else popup.classList.add("hidden");
}

// Dynamic Sidebar Page Management (Supports Chat & Image)
function renderSidebar() {
  const list = document.getElementById("saved-pages");
  if (!list) return;

  list.innerHTML = "";

  savedPages.forEach((page, index) => {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.href = "#";
    link.className = `page-link ${activePageIndex === index ? 'active' : ''}`;
    
    const icon = page.type === 'image' ? '🎨' : '💬';
    link.innerHTML = `<span>${icon} ${page.title}</span>`;

    link.onclick = (e) => {
      e.preventDefault();
      loadPage(index);
    };

    link.oncontextmenu = (e) => {
      e.preventDefault();
      e.stopPropagation();
      showContextMenu(e.clientX, e.clientY, index);
    };

    li.appendChild(link);
    list.appendChild(li);
  });
}

function startNewSession() {
  activePageIndex = null;
  document.getElementById("chat-window").innerHTML = "";
  document.getElementById("image-gallery-window").innerHTML = `
    <div class="bot-msg initial-image-prompt">
      ✨ Welcome to Image Studio! Enter a prompt below to generate custom images.
    </div>`;
  renderSidebar();
}

function saveNewPageSession(promptText, type = 'chat') {
  if (activePageIndex === null) {
    let pageTitle = promptText.length > 20 ? promptText.substring(0, 18) + "..." : promptText;
    const content = type === 'image' 
      ? document.getElementById("image-gallery-window").innerHTML 
      : document.getElementById("chat-window").innerHTML;

    savedPages.push({
      title: pageTitle,
      type: type,
      content: content
    });

    activePageIndex = savedPages.length - 1;
    localStorage.setItem("savedPages", JSON.stringify(savedPages));
    renderSidebar();
  }
}

function updateCurrentPageSession() {
  if (activePageIndex !== null && savedPages[activePageIndex]) {
    const type = savedPages[activePageIndex].type;
    savedPages[activePageIndex].content = type === 'image'
      ? document.getElementById("image-gallery-window").innerHTML
      : document.getElementById("chat-window").innerHTML;
    
    localStorage.setItem("savedPages", JSON.stringify(savedPages));
  }
}

function showContextMenu(x, y, index) {
  let contextMenu = document.getElementById("context-menu");
  
  if (!contextMenu) {
    contextMenu = document.createElement("div");
    contextMenu.id = "context-menu";
    contextMenu.className = "context-menu";
    document.body.appendChild(contextMenu);
  }

  contextMenu.innerHTML = `<div class="menu-item delete-item" onclick="deletePage(${index})">🗑️ Delete Page</div>`;
  contextMenu.style.top = `${y}px`;
  contextMenu.style.left = `${x}px`;
  contextMenu.style.display = "block";
}

function deletePage(index) {
  savedPages.splice(index, 1);
  localStorage.setItem("savedPages", JSON.stringify(savedPages));

  if (activePageIndex === index) {
    startNewSession();
  } else if (activePageIndex > index) {
    activePageIndex--;
    renderSidebar();
  } else {
    renderSidebar();
  }

  const contextMenu = document.getElementById("context-menu");
  if (contextMenu) contextMenu.style.display = "none";
}

function loadPage(index) {
  activePageIndex = index;
  const page = savedPages[index];
  if (!page) return;

  if (page.type === 'image') {
    switchWorkspace('image');
    document.getElementById("image-gallery-window").innerHTML = page.content;
  } else {
    switchWorkspace('chat');
    document.getElementById("chat-window").innerHTML = page.content;
  }
  renderSidebar();
}

function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  const chatWindow = document.getElementById("chat-window");
  
  const userMsg = document.createElement("div");
  userMsg.className = "user-msg";
  userMsg.textContent = message;
  chatWindow.appendChild(userMsg);

  saveNewPageSession(message, 'chat');

  fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: message })
  })
    .then(res => res.json())
    .then(data => {
      const botMsg = document.createElement("div");
      botMsg.className = "bot-msg";
      botMsg.innerHTML = formatBotResponse(data.reply);

      chatWindow.appendChild(botMsg);
      chatWindow.scrollTop = chatWindow.scrollHeight;
      updateCurrentPageSession();
    })
    .catch(err => console.error("Error sending message:", err));

  input.value = "";
}