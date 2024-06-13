using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AustinHarris.JsonRpc;
using System.Reflection;
using System;

class Params{
    public float[] angles;
    // list of strings
    public string[] jointNames;
    public string handName;

    public Params(float[] angles, string[] jointNames, string handName){
        this.angles = angles;
        this.jointNames = jointNames;
        this.handName = handName;
    }
    public float GetAttr(string name)
    {
        Type type = this.GetType();
        PropertyInfo propertyInfo = type.GetProperty(name);

        if (propertyInfo != null)
        {
            // Get the value of the property
            object value = propertyInfo.GetValue(this);

            // Convert the value to float if possible
            if (value is float floatValue)
            {
                return floatValue;
            }
            else if (value is bool boolValue)
            {
                return boolValue ? 1 : 0 ;
            }
            // You might want to handle other types here if needed
            // For simplicity, let's throw an exception if the value isn't a float
            else
            {
                throw new InvalidOperationException("Property value is not a float.");
            }
        }

        throw new ArgumentException("Property not found!");
    }

}
public class Hands : MonoBehaviour
{

    class Rpc: JsonRpcService
    {
        Hands hands;
        GameObject handLabel;
        GameObject handPrediction;
        Transform[] children;
        public Rpc(Hands hands)
        {
            this.hands = hands;
        }

        [JsonRpcMethod]
        void UpdateLeapHands(Params angles)
        {
            // bool pred = false;
            if (angles.handName == "Label")
            {
                handLabel = hands.FindChildGameObject("HandLabel_1");
                children = handLabel.GetComponentsInChildren<Transform>();
            }
            else
            {
                handPrediction = hands.FindChildGameObject("HandPrediction_1");
                children = handPrediction.GetComponentsInChildren<Transform>();
                // pred = true;
            }
            foreach (Transform child in children)
            {   

                // get the name of the child
                string name = child.name;
                if (name.Contains("bone") || name.EndsWith("_1"))
                {
                    continue;
                }
                // if name not in jointNames array skip
                // if (!angles.jointNames.Contains(name + "_Flex") && !angles.jointNames.Contains(name + "_Adb") ){
                //     continue;
                // }
                //  if name is not thumb and containes tmc skip
                if (!name.Contains("Thumb") && name.Contains("TMC")){
                    continue;
                }

                // if (name.Contains("Thumb")){
                //     continue;
                // }
                
                float[] fingerAngles = new float[3];
                int index_flex = -1;
                float val_flex = 0.0f;

                try
                {
                    index_flex = Array.IndexOf(angles.jointNames, name + "_Flex");
                    if (index_flex != -1)
                    {
                        val_flex = angles.angles[index_flex];
                    }
                    else
                    {
                        Console.WriteLine("Name not found in jointNames array");
                    }
                }
                catch (IndexOutOfRangeException e)
                {
                    Console.WriteLine("Error: " + e.Message);
                }
                  
                float val_abd = child.localEulerAngles.z;
                Debug.Log(child.name);
                if (name.Contains("TMC") || name.Contains("MCP")){
                    int index_abd = Array.IndexOf(angles.jointNames, name+"_Adb");
                    val_abd = angles.angles[index_abd];
                }

                fingerAngles[0] = child.localEulerAngles.x;
                fingerAngles[1] = -val_flex;
                fingerAngles[2] = child.localEulerAngles.z;

                
                if (name.Contains("Thumb") ){
                    
                    // fingerAngles[1] = val_flex;
                    fingerAngles[1] = child.localEulerAngles.y;
                    fingerAngles[2] = -val_flex;

                    if (name.Contains("TMC")){
                            fingerAngles[2] += 42; 
                    }
                }
                // set the angles of the finger
                child.localEulerAngles = new Vector3(fingerAngles[0], fingerAngles[1], fingerAngles[2]);
            }
        }
        [JsonRpcMethod]
        void UpdateHand(Params angles)
        {

            if (angles.handName == "Label")
            {
                handLabel = hands.FindChildGameObject("HandLabel_1");
                children = handLabel.GetComponentsInChildren<Transform>();
            }
            else
            {
                handPrediction = hands.FindChildGameObject("HandPrediction_1");
                children = handPrediction.GetComponentsInChildren<Transform>();
            }
            
        
            foreach (Transform child in children)
            {   
                // get the name of the child
                string name = child.name;
                if (name.Contains("bone") || name.EndsWith("_1"))
                {
                    continue;
                }
                // Debug.Log(name);
                float[] fingerAngles = new float[3];
                if (name.Contains("CMC") || name.Contains("MCP")){

                    int index_flex = Array.IndexOf(angles.jointNames, name+"_Flex");
                    int index_abd = Array.IndexOf(angles.jointNames, name+"_Spread");
                            

                    //  if index is -1, then the joint is not in the list therefore it is not flexed
                    if (index_flex == -1){
                        fingerAngles[0] = child.localEulerAngles.x;
                        fingerAngles[1] = child.localEulerAngles.y;
                        fingerAngles[2] = 45+angles.angles[index_abd];
                    }
                    else if (index_abd == -1){
                        fingerAngles[0] = child.localEulerAngles.x;
                        fingerAngles[1] = -angles.angles[index_flex];
                        fingerAngles[2] = child.localEulerAngles.z;
                    }
                    else{
                        fingerAngles[0] = child.localEulerAngles.x;
                        fingerAngles[1] = -angles.angles[index_flex];
                        fingerAngles[2] = 45+angles.angles[index_abd];
                    }
                    
                }
                else{
                    int index_flex = Array.IndexOf(angles.jointNames, name+"_Flex");               
                    if (index_flex == -1){
                        fingerAngles[0] = child.localEulerAngles.x;
                        fingerAngles[1] = child.localEulerAngles.y;
                        fingerAngles[2] = child.localEulerAngles.z;
                    }
                    else{
                        fingerAngles[0] = child.localEulerAngles.x;
                        fingerAngles[1] = -angles.angles[index_flex];
                        fingerAngles[2] = child.localEulerAngles.z;
                    }
                    if (name.Contains("Thumb")){
                        fingerAngles[0] = child.localEulerAngles.x;
                        fingerAngles[1] = child.localEulerAngles.y;
                        fingerAngles[2] = -angles.angles[index_flex];
                    }
                }

                // set the angles of the finger
                child.localEulerAngles = new Vector3(fingerAngles[0], fingerAngles[1], fingerAngles[2]);
            }


        }
       
    }
    public GameObject FindChildGameObject(string childName)
    {
        Transform childTransform = this.transform.Find(childName);

        if (childTransform != null)
        {
            Debug.Log("Child object found");
            return childTransform.gameObject;
            
        }
        else
        {
            Debug.Log("Child object not found");
            return null;
        }
    }
    Rpc rpc;
    // Start is called before the first frame update
    void Start()
    {
        rpc = new Rpc(this);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
