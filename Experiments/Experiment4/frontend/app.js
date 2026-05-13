const healthStatus = document.querySelector("#healthStatus");
const personaSelect = document.querySelector("#personaSelect");
const personaName = document.querySelector("#personaName");
const personaProblem = document.querySelector("#personaProblem");
const personaDetails = document.querySelector("#personaDetails");
const personaSolution = document.querySelector("#personaSolution");
const runButton = document.querySelector("#runButton");
const runStatus = document.querySelector("#runStatus");
const saveResults = document.querySelector("#saveResults");
const keepMemory = document.querySelector("#keepMemory");
const problemStatement = document.querySelector("#problemStatement");
const patternsList = document.querySelector("#patternsList");
const rootCausesList = document.querySelector("#rootCausesList");
const recommendationsList = document.querySelector("#recommendationsList");
const filesList = document.querySelector("#filesList");

let personas = [];

function setStatus(element, text, className = "") {
  element.textContent = text;
  element.className = element.className
    .split(" ")
    .filter((name) => !["ok", "error"].includes(name))
    .concat(className ? [className] : [])
    .join(" ");
}

function fillList(element, items, renderItem = (item) => item) {
  element.innerHTML = "";
  if (!items || items.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No items yet.";
    element.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = renderItem(item);
    element.appendChild(li);
  });
}

function renderPersona(row) {
  const persona = personas.find((item) => item.row === Number(row));
  if (!persona) {
    return;
  }

  personaName.textContent = persona.name;
  personaProblem.textContent = persona.problem;
  personaDetails.textContent = persona.person_details;
  personaSolution.textContent = persona.ethnographic_solution;
}

async function loadPersonas() {
  const response = await fetch("/api/personas");
  if (!response.ok) {
    throw new Error(await response.text());
  }

  const data = await response.json();
  personas = data.personas;
  personaSelect.innerHTML = "";

  personas.forEach((persona) => {
    const option = document.createElement("option");
    option.value = persona.row;
    option.textContent = `${persona.row}. ${persona.name}`;
    personaSelect.appendChild(option);
  });

  renderPersona(personas[0]?.row);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("API unavailable");
    }
    setStatus(healthStatus, "API ready", "ok");
  } catch (error) {
    setStatus(healthStatus, "API offline", "error");
  }
}

function renderSession(data) {
  const artifact = data.artifact;
  const agent1 = artifact.agent1 || {};
  const agent3 = artifact.agent3 || {};

  problemStatement.textContent =
    agent1.final_problem_statement || "No problem statement returned.";
  fillList(patternsList, agent3.patterns);
  fillList(rootCausesList, agent3.root_causes);
  fillList(recommendationsList, agent3.recommendations, (item) => {
    const rank = item.rank ? `Rank ${item.rank}: ` : "";
    const reason = item.reason ? ` (${item.reason})` : "";
    return `${rank}${item.recommendation}${reason}`;
  });

  const fileEntries = Object.entries(data.files || {}).map(([label, path]) => `${label}: ${path}`);
  fillList(filesList, fileEntries);
}

async function runSession() {
  runButton.disabled = true;
  runButton.textContent = "Running...";
  runStatus.textContent = "Agents are working";

  try {
    const response = await fetch("/api/sessions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        row: Number(personaSelect.value),
        keep_memory: keepMemory.checked,
        save_results: saveResults.checked,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Session failed");
    }

    renderSession(await response.json());
    runStatus.textContent = "Complete";
  } catch (error) {
    runStatus.textContent = "Failed";
    problemStatement.textContent = error.message;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run session";
  }
}

personaSelect.addEventListener("change", (event) => renderPersona(event.target.value));
runButton.addEventListener("click", runSession);

checkHealth();
loadPersonas().catch((error) => {
  personaProblem.textContent = error.message;
  setStatus(healthStatus, "API error", "error");
});
