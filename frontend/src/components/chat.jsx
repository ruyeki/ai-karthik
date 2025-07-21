import { Input } from "@heroui/input";
import { Button } from "@heroui/button";
import { Form } from "@heroui/form";
import { useState } from "react";
import { Card } from "@heroui/card";
import { User } from "@heroui/react";  // removed unused 'image'
import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";
import { TypingIndicator } from "@chatscope/chat-ui-kit-react";
import ReactMarkdown from "react-markdown";
import { TypeAnimation } from "react-type-animation";
import { Image } from "@heroui/image";
import {Spinner} from "@heroui/react";
import {Progress} from "@heroui/progress";

export default function Chat({projectName, messages, setMessages}) {
  const [userText, setUserText] = useState("");
  const [isLoading, setLoading] = useState(false);
  const [images, setImages] = useState([]);
  const [formDisabled, setFormDisabled] = useState(false);
  const [fileList, setFileList] = useState([]);


  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!userText.trim()) return; //if user input is blank do nothing

    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: userText }]);
    setUserText("");
    setImages([]);

    const formData = new FormData();
    formData.append("question", userText);
    formData.append("project_name", projectName);

    for(let i = 0; i<fileList.length; ++i){
      formData.append("pdf", fileList[i]);
    }

    console.log("Files in formData:");
    for (let [key, value] of formData.entries()) {
      console.log(key, value.name || value);
    }



    try {
      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        body: formData,
      });

      const chatResponse = await response.json();
      console.log(chatResponse);

      let botMessage = chatResponse.text_response || "Something went wrong. Try again.";

      if(chatResponse.doc_url){
          botMessage += `\n\n\n\n[DOWNLOAD REPORT](${chatResponse.doc_url})`;
      }

      const botImages = Array.isArray(chatResponse.images) ? chatResponse.images : [];

      console.log("These are the images ", botImages);

      //this is really hacky, probably need to come up with something better
      //const shouldIncludeImage = userText.toLowerCase().includes("image") || userText.toLowerCase().includes("summar") || userText.toLowerCase().includes("report") || userText.toLowerCase().includes("table") || userText.toLowerCase().includes("graph") || userText.toLowerCase().includes("data") || userText.toLowerCase().includes("visual");

      setImages(botImages);

      setMessages((prev) => [
        ...prev,
        {
          role: "chatbot",
          content: botMessage,
          images: botImages,
        },
      ]);
    } catch (error) {
      console.error("Error sending user query:", error);
      setMessages((prev) => [
        ...prev,
        { role: "chatbot", content: "Damn, something went wrong. Try again!" },
      ]);
    }

    setLoading(false);
    setFileList([]);
  };

  // Group messages in pairs: user + bot
  const groupedMessages = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === "user") {
      if (messages[i + 1] && messages[i + 1].role === "chatbot") {
        groupedMessages.push([messages[i], messages[i + 1]]);
        i++;
      } else {
        groupedMessages.push([messages[i]]);
      }
    } else {
      groupedMessages.push([messages[i]]);
    }
  }

  return (
    
    <div className="flex flex-col h-[95vh] bg-white">

      {/* Chat messages section */}
      <div className="flex-1 overflow-y-auto px-4 py-5 max-w-8xl mx-auto w-full">
        {groupedMessages.length === 0 && !projectName &&(
          <div>
            <h1 className="text-4xl text-center absolute inset-0 flex items-center justify-center pointer-events-none select-none ms-60 mb-20">
              <TypeAnimation
                sequence={["Welcome.", 1500, "Select a project to get started.", 1300]} 
              />
            </h1>
          </div>
        )}

        {projectName && groupedMessages.length === 0 && (
          <div>
            <h1 className="text-4xl text-center absolute inset-0 flex items-center justify-center pointer-events-none select-none ms-60 mb-20">
              <TypeAnimation
                sequence={[`Welcome to ${projectName}.`, 1300]} 
              />
            </h1>
          </div>
        )}
        
        {groupedMessages.map((pair, idx) => (
          <div key={idx}>
            {pair.map((message, index) => {
              const isUser = message.role === "user";
              return (
                <div key={index} className="flex items-start gap-2 mb-3">
                  {!isUser && (
                    <User
                      avatarProps={{
                        src: "https://media.licdn.com/dms/image/v2/C5603AQHOSz9IYYlDQw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1630514668239?e=2147483647&v=beta&t=5gxGeIgoopocSdhFM7497-2w7RMRUXQydvAsQGtoBm0",
                      }}
                      className="flex-shrink-0"
                      isFocusable={true}
                    />
                  )}
                  <Card
                    className={`p-3 mb-3 w-full break-words ${
                      isUser ? "bg-blue-50" : "bg-gray-50"
                    }`}
                  >
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                    {message.images && message.images.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {message.images.map((img, index) => (
                          <Image key={index} src={img} />
                        ))}
                      </div>
                    )}
                      </Card>
                </div>
              );
            })}
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 text-gray-500 italic mb-3">
            <User
              avatarProps={{
                src: "https://media.licdn.com/dms/image/v2/C5603AQGqnSJy_C5s_w/profile-displayphoto-shrink_200_200/0/1549603008865?e=2147483647&v=beta&t=mJOD2T5cgGPrRu_j-Knr-cbQcVRVuaJgDUWuG_XI7PE",
              }}
              className="flex-shrink-0"
              isFocusable={true}
            />
            <Spinner variant="dots" />
          </div>
        )}
      </div>
      <Form onSubmit={handleSubmit} className="mt-5" encType="multipart/form-data">
        <div className="flex items-center gap-2 max-w-6xl mx-auto w-full mt-5">

          <Input 
            type="text"
            className="flex-1 h-10"
            value={userText}
            onChange={(e) => setUserText(e.target.value)}
            name="user-input"
            size="lg"
            placeholder="Ask AI Karthik anything..."
            disabled={!projectName || formDisabled}
          />
          <label
            htmlFor="fileUpload"
            className="cursor-pointer w-12 h-12 flex items-center justify-center rounded-full hover:bg-primary text-black shadow-lg transition mt-3 border-2 border-primary"
            title="Upload File"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="blue"
              strokeWidth="2"
              
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
          </label>

          <Input
            id="fileUpload"
            type="file"
            accept=".pdf"
            name = "pdf"
            multiple
            className="hidden"
            onChange={(e) => setFileList(e.target.files)}
          />
          <Button
            type="submit"
            color="primary"
            aria-label="Send"
            className="rounded-full flex items-center justify-center mb-[-10px]"
            radius="full"
            isLoading={isLoading}
            size="lg"
            variant="ghost"
            disabled={!projectName || formDisabled}
          >
            {!isLoading && (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            )}
          </Button>

          
        </div>
      </Form>
      <p className="justify-center text-center text-xs mt-5 mr-20 ms-[-20]">
        AI Karthik can make mistakes. Check important info.
      </p>
    </div>
  );
}