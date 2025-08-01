import Chat from "./components/chat";
import MySidebar from "./components/sidebar";
import React, { useState } from "react";

export default function App() {
      const [selectedProject, setSelectedProject] = useState("");
      const [messages, setMessages] = useState([]);
  
      const onSelectProject = (projectName) => {
          setSelectedProject(projectName);
          setMessages([]);
      }
  
  return (
    <div>
      <div className="flex h-screen">
        <div className="w-70 border-r border-gray-200">
          <MySidebar onSelectProject = {onSelectProject} />
        </div>
        <div className="flex-1 p-4 overflow-auto">
          <Chat projectName={selectedProject} messages = {messages} setMessages = {setMessages} />
        </div>
      </div>
    </div>
  );
}