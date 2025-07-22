import React from "react";
import { Sidebar as ProSidebar, Menu, MenuItem } from "react-pro-sidebar";
import { Button, Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure, Input, Form } from "@heroui/react";
import {Image} from "@heroui/react";
import { useState, useEffect } from "react";
import {Progress} from "@heroui/react";
import {Spinner} from "@heroui/react";


export default function MySidebar({onSelectProject}) {

  const { isOpen, onOpen, onClose } = useDisclosure();
  const [backdrop, setBackdrop] = React.useState("opaque");
  const [userInput, setUserInput] = useState("");
  const [isLoading, setLoading] = useState(false);
  const [projectNames, setProjectNames] = useState([]);
  const [activeProject, setActiveProject] = useState("");
  const [activeProjectLoading, setActiveProjectLoading] = useState(false);
  const [sideBarLoading, setSideBarLoading] = useState(false);
  const backdrops = ["opaque", "blur", "transparent"];

  useEffect(()=>{
    getProjects();
  }, []);

  const handleOpen = (backdropType) => {
    setBackdrop(backdropType);
    onOpen();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) {
    console.warn("Project name is empty — not submitting.");
    return;
  }
    setLoading(true);

    try{
      const response = await fetch("http://127.0.0.1:5000/create_project", {
          method: "POST", 
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({project_name: userInput})
      });

      const data = await response.json();

      setUserInput("");

    }catch(error){
      console.error(error);
    }

    setLoading(false);
    onClose();
  }

  const handleSideBarClick = async (projectName) =>{
    setActiveProjectLoading(true);

    setLoading(true);
    setActiveProject(projectName);
    onSelectProject(projectName);

    try{
      const response = await fetch("http://127.0.0.1:5000/connect_to_db", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({project_name: projectName})
      });

      const data = await response.json()
      console.log(data);


    }catch (error){
      console.error(error);
    }finally{

    setLoading(false);
      setTimeout(() => {
      setActiveProjectLoading(false);
    }, 500); 
    }
  }

  const getProjects = async() => {

    setSideBarLoading(true);

    try{
      const response = await fetch("http://127.0.0.1:5000/get_project_names",{
        method: "GET"
      });

      if(!response.ok){
        throw new Error(`HTTP error. Status: ${response.status}`);
      }

      const data = await response.json();
      console.log(data);

      setProjectNames(data.projects);

    }catch (error){
      console.error(error);
    }finally{
      setSideBarLoading(false);
    }
  }

  return (
    <div className="flex">
      <ProSidebar
        style={{
          height: "100vh",
          backgroundColor: "#f9fafb",
          borderRight: "1px solid #e5e7eb",
          justifyContent: "space-between",
        }}
      >
      <div className="overflow-y-auto">
      
      {activeProjectLoading&& (
      <Progress isIndeterminate aria-label="Loading..." className="sticky top-0 max-w-md" size="sm" />
      )}

        {sideBarLoading ? (
          <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
            <Spinner size="lg" />
          </div>
        ) : (
      <Menu iconShape="circle">
      {projectNames.map((project, index) => (
          <MenuItem style={menuItemStyle} onClick={() => handleSideBarClick(project.name)} active = {activeProject === project.name}>
            {project.name}
          </MenuItem>
      ))}
      </Menu>
  )}
      </div>

        <div className="sticky bottom-0 bg-[#f9fafb] p-4 flex flex-col gap-3 ">

          <Button key = "blur" className = "capitalize" variant = "ghost" onPress={() => handleOpen("blur")}>Add Project</Button>

        </div>

      </ProSidebar>

      <Modal backdrop={backdrop} isOpen={isOpen} onClose={onClose}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader className="flex flex-col gap-1">
                Add Project
              </ModalHeader>
              <ModalBody>
                <Form onSubmit={handleSubmit}>
                  <h1>Project Name: </h1>
                  <Input placeholder = "Enter a project name" value = {userInput} onChange = {(e) => setUserInput(e.target.value)}></Input>
                </Form>
              </ModalBody>
              <ModalFooter>
                <Button color="danger" variant="light" onPress={onClose}>
                  Close
                </Button>
                <Button isLoading={isLoading} type = "submit" color="primary" onPress={onClose}>
                  Submit
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
  );
}


const menuItemStyle = {
  padding: '10px 15px',
  borderRadius: '0.375rem', 
  margin: '4px 8px',
  transition: 'background 0.2s ease',
  fontSize: '0.95rem',
  color: '#374151', 
};

const ButtonStyle = {
  padding: '10px 15px',
  borderRadius: '0.375rem', 
  margin: '4px 8px',
  fontSize: '0.95rem',
};