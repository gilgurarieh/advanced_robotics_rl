clc
clear all
close all

sim=remApi('remoteApi');
sim.simxFinish(-1);
clientID=sim.simxStart('127.0.0.1',19999,true,true,5000,5);
%sim.simxSynchronous(clientID,true);
%sim.simxStartSimulation(clientID,sim.simx_opmode_oneshot_wait);


if (clientID>-1)
    disp('Connecton to remote API server Established');

    [~,dum]= sim.simxGetObjectHandle(clientID,'Quadricopter_target',sim.simx_opmode_blocking);

    % Get the initial location and orientation of the quadrotor
    [returnCode,Int_Quad_Poss] = sim.simxGetObjectPosition(clientID,dum,-1,sim.simx_opmode_blocking); % In meter, vecor
    [returnCode,Int_Quad_Ortt] = sim.simxGetObjectOrientation(clientID,dum,-1,sim.simx_opmode_blocking); % Ret in radian, vector
    [returnCode,Int_Quad_Quat] = sim.simxGetObjectQuaternion(clientID,dum,-1,sim.simx_opmode_oneshot_wait);  

    i = 1;
    while(i < 4)        

        % Set object target position
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1) Int_Quad_Poss(2) Int_Quad_Poss(3)],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1)+0.5 Int_Quad_Poss(2) Int_Quad_Poss(3)],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1) Int_Quad_Poss(2) Int_Quad_Poss(3)],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1) Int_Quad_Poss(2)+0.5 Int_Quad_Poss(3)],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1) Int_Quad_Poss(2) Int_Quad_Poss(3)],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1) Int_Quad_Poss(2) Int_Quad_Poss(3)+0.5],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1) Int_Quad_Poss(2) Int_Quad_Poss(3)],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1)+0.5 Int_Quad_Poss(2)+0.5 Int_Quad_Poss(3)+0.5],sim.simx_opmode_blocking);
        pause(2)
        [returnCode] = sim.simxSetObjectPosition(clientID,dum,-1,[Int_Quad_Poss(1) Int_Quad_Poss(2) Int_Quad_Poss(3)],sim.simx_opmode_blocking);
        pause(4)

        % The lua scripts only supoert Yaw rotation change
        % Set object target orientation using Euler angles
        [returnCode] = sim.simxSetObjectOrientation(clientID,dum,-1,[Int_Quad_Ortt(3) Int_Quad_Ortt(3) Int_Quad_Ortt(3)+2],sim.simx_opmode_oneshot);
        pause(4)
        [returnCode] = sim.simxSetObjectOrientation(clientID,dum,-1,[Int_Quad_Ortt(3) Int_Quad_Ortt(3) Int_Quad_Ortt(3)],sim.simx_opmode_oneshot);
        pause(4)
        [returnCode] = sim.simxSetObjectOrientation(clientID,dum,-1,[Int_Quad_Ortt(3) Int_Quad_Ortt(3) Int_Quad_Ortt(3)-2],sim.simx_opmode_oneshot);
        pause(2)
        [returnCode] = sim.simxSetObjectOrientation(clientID,dum,-1,Int_Quad_Ortt,sim.simx_opmode_oneshot);
        % Set object target orientation using Quaternions  
        pause(2)
        [returnCode] = sim.simxSetObjectQuaternion(clientID,dum,-1,Int_Quad_Quat,sim.simx_opmode_oneshot_wait);
        pause(4)
        [returnCode] = sim.simxSetObjectQuaternion(clientID,dum,-1,[0 0 0.8415 0.5403],sim.simx_opmode_oneshot_wait);
        
        % Send Quaternion value to Coppliasim
        pause(2)
        Send_Quaternion = [0 0 0.8415 0.5403];
        packedData=sim.simxPackFloats(Send_Quaternion);
        sim.simxSetStringSignal(clientID,'Quaternion_Matlab',packedData,sim.simx_opmode_oneshot);
        pause(4)
        Send_Quaternion = [0 0 0 1];
        packedData=sim.simxPackFloats(Send_Quaternion);
        sim.simxSetStringSignal(clientID,'Quaternion_Matlab',packedData,sim.simx_opmode_oneshot);            

        %i = i + 1;

    end
    sim.simxStopSimulation(clientID,sim.simx_opmode_oneshot_wait);
    sim.simxFinish(-1); % Ending the connection (clientID) refer to specific simxStart. Can be -1 to end all running communication threads.
else
    fprintf('Connection Failed...\nRetrying\n');
end
[returnCode] = sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait); % Stopping Simulation
[returnCode] = sim.simxCloseScene(clientID, sim.simx_opmode_oneshot_wait); % Closing the loaded V-Rep scene 
sim.delete();
fprintf('Simulation Ended\n');