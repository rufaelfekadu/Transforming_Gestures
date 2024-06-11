using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AustinHarris.JsonRpc;


public class HandPrediction : MonoBehaviour
{

    class Rpc: JsonRpcService
    {
        HandPrediction hand;
        public Rpc(HandPrediction hand)
        {
            this.hand = hand;
        }

        [JsonRpcMethod]
        void UpdateHand(float[] angles)
        {
            // get the child objects of the hand
            Transform[] children = hand.GetComponentsInChildren<Transform>();
            // update the angles of the children
            int i = 0;

            foreach (Transform child in children)
            {   
                // get the name of the child
                string name = child.name;
                // if the name is not a finger, skip it
                if (!name.Contains("finger") || name.EndsWith("_1"))
                {
                    continue;
                }
                // get the angles of the finger
                float[] fingerAngles = new float[3];
                fingerAngles[0] = child.localEulerAngles.x;
                fingerAngles[1] = -angles[i++];
                fingerAngles[2] = child.localEulerAngles.z;

                // set the angles of the finger
                child.localEulerAngles = new Vector3(fingerAngles[0], fingerAngles[1], fingerAngles[2]);
            }

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
