import { Input } from "@heroui/input";
import { Button } from "@heroui/button";
import { Form } from "@heroui/form";
import { useState } from "react";
import { Card } from "@heroui/card";
import { User } from "@heroui/react";
import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";
import { TypingIndicator } from "@chatscope/chat-ui-kit-react";

export default function Chat() {
  const [userText, setUserText] = useState("");
  const [chatText, setChatText] = useState("");
  const [isLoading, setLoading] = useState(false); 

  const handleSubmit = async (e) => {
    e.preventDefault();
    setUserText("");

    if (!userText.trim()) return; //if user clicks enter and input is blank, nothing happens

    setLoading(true); // start loading indicator

    try {
      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userText }),
      });

      const chatResponse = await response.json();
      setChatText(chatResponse.text_response || "Something went wrong. Try again.");
    } catch (error) {
      console.error("Error sending user query:", error);
      setChatText("Oops! Something went wrong.");
    }
    setLoading(false); // stop loading
  };

  return (
    <div className="fixed bottom-0 left-0 w-full bg-white p-4 shadow-md">
      <div className="max-h-80 overflow-y-auto max-w-3xl mx-auto mb-20">
        {isLoading && (
          <div
            style={{
              margin: "10px 0",
              alignSelf: "flex-start",
              color: "#888",
              fontStyle: "italic",
            }}
            className="flex items-center gap-2"
          >
            <User
              avatarProps={{
                src: "https://media.licdn.com/dms/image/v2/C5603AQGqnSJy_C5s_w/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1549603008865?e=2147483647&v=beta&t=mJOD2T5cgGPrRu_j-Knr-cbQcVRVuaJgDUWuG_XI7PE",
              }}
              className="flex-shrink-0"
              isFocusable={true}
            />
            <TypingIndicator  />
          </div>
        )}

        {chatText && (
          <div className="flex items-start gap-2 mb-3">
            <User
              avatarProps={{
                src: "https://media.licdn.com/dms/image/v2/C5603AQGqnSJy_C5s_w/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1549603008865?e=2147483647&v=beta&t=mJOD2T5cgGPrRu_j-Knr-cbQcVRVuaJgDUWuG_XI7PE",
              }}
              className="flex-shrink-0"
              isFocusable={true}
            />
            <Card className="p-3 text-sm text-gray-800 rounded-lg bg-gray-100 flex-1 shadow-sm">
              <p className="whitespace-pre-wrap">{chatText}</p>
            </Card>
          </div>
        )}
      </div>

      <Form onSubmit={handleSubmit}>
        <div className="flex items-center gap-2 max-w-3xl mx-auto w-full">
          <Input
            type="text"
            placeholder="Ask anything..."
            className="flex-1"
            value={userText}
            onChange={(e) => setUserText(e.target.value)}
            name="user-input"
          />
          <Button type="submit" color="primary" aria-label="Send">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="currentColor"
              className="w-5 h-5 scale-x-[-1]"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3.75 12L20.25 3.75 16.5 12l3.75 8.25L3.75 12z"
              />
            </svg>
          </Button>
        </div>
      </Form>
    </div>
  );
}
