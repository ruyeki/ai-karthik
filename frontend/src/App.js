import {Input} from "@heroui/react";
import Chat from "./components/chat";

export default function App() {
  return (
    <div>
      <h1 className = "text-center justify-center text-4xl font-semibold">Project Report Generator</h1>
      <Chat />
    </div>
  );
}