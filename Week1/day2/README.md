# Day 2 - System Role & Temperature in LLM

## 📌 Objective

Learn how to control an LLM's behavior using **System Messages** and understand the effect of the **Temperature** parameter on responses.

---

## 📂 Project Structure

```
day2/
├── main.py
├── README.md
├── Day2-Notes.jpg
├── pyproject.toml
├── uv.lock
├── .env          # Ignored
└── .venv/        # Ignored
```

---

## 🛠️ Commands Used

### Create Virtual Environment

```bash
uv venv
```

### Activate Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
uv add groq python-dotenv
```

### Run the Program

```bash
python main.py
```

### Deactivate Virtual Environment

```bash
deactivate
```

---

## 📦 Packages Used

- groq
- python-dotenv

---

## 🔑 Environment Variable

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_api_key_here
```

---

## 💻 What the Code Does

- Loads the API key securely from the `.env` file.
- Creates a Groq client.
- Uses the **Llama 3.3 70B Versatile** model.
- Sends a **System Message** to define the AI's behavior.
- Sends a **User Message** with the actual prompt.
- Uses **Temperature = 0** for deterministic responses.
- Prints the generated company name.

---

## 🧠 Concepts Learned

- System Role
- User Role
- Prompt Engineering Basics
- Temperature
- Message Order
- Context-Based Conversation
- Chat Completion API

---

## 📖 Key Learnings

### System Message

Defines the AI's behavior before answering.

Example:

> "You are a brand manager who suggests company names."

---

### User Message

Contains the actual request from the user.

Example:

> "Suggest a name for my clothing company."

---

### Temperature

Controls randomness in responses.

- **0** → Consistent and deterministic responses.
- **1** → Balanced creativity.
- **2** → Highly creative and random.

---

## 📝 Handwritten Notes

Topics covered:

- Virtual Environment
- API Key
- Client
- Messages
- Roles (System, User, Assistant)
- Temperature
- Context-Based Conversation
- LLM Response Flow

![Day 2 Notes](Day2-Notes.jpg)

---

## ✅ Outcome

Successfully controlled the LLM's behavior using a System Message and generated deterministic responses using the Temperature parameter.