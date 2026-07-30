#Tello Drone Command made by Drone Academy Thailand that handles drone setup. Adapted by North Athith Ittiphakorn for personal use.

from djitellopy import Tello
import cv2
import os
import platform
import subprocess
import sys
import threading
import time


def get_current_ssid() -> str | None:
    if platform.system() != "Windows":
        return None
    try:
        output = subprocess.check_output(
            ["netsh", "wlan", "show", "interfaces"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return None

    for line in output.splitlines():
        if line.strip().startswith("SSID") and ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == "SSID":
                return value.strip()
    return None

# QR scanning is optional. Some Windows installs of pyzbar fail because the
# native zbar libraries are missing, so import it defensively.
try:
    from pyzbar.pyzbar import decode
except Exception as e:
    decode = None
    PYZBAR_IMPORT_ERROR = e
else:
    PYZBAR_IMPORT_ERROR = None

#import zxingcpp

"""
DroneTello - Enhanced Tello Drone Controller

BASIC USAGE:
    # Simple connection and flight
    drone = DroneTello()
    drone.takeoff()
    drone.wait(2)
    drone.land()
    drone.cleanup()

CAMERA USAGE:
    # With live camera display
    drone = DroneTello(show_cam=True)
    drone.takeoff()
    drone.capture("my_photo.jpg")           # Take picture
    data = drone.scan_qr("my_photo.jpg")    # Read QR code
    drone.land()
    drone.cleanup()

MISSION PAD USAGE:
    # With mission pad detection
    drone = DroneTello(enable_mission_pad=True)
    drone.takeoff()
    # Use mission pad commands like go_xyz_speed_mid()
    drone.land()
    drone.cleanup()

MOVEMENT COMMANDS (inherited from Tello):
    drone.move_up(50)        # Move up 50cm
    drone.move_down(50)      # Move down 50cm
    drone.move_forward(50)   # Move forward 50cm
    drone.move_back(50)      # Move backward 50cm
    drone.move_left(50)      # Move left 50cm
    drone.move_right(50)     # Move right 50cm
    drone.rotate_clockwise(90)        # Rotate 90 degrees clockwise
    drone.rotate_counter_clockwise(90) # Rotate 90 degrees counter-clockwise
"""

class DroneTello(Tello):
    """
    Enhanced Tello drone class with camera display and mission pad support.
    
    Usage:
        drone = DroneTello(show_cam=True, enable_mission_pad=True)
        drone.takeoff()
        drone.capture("photo.jpg")
        data = drone.scan_qr("photo.jpg")
        drone.land()
    """
    def __init__(self, show_cam=False, enable_mission_pad=False):
        """
        Initialize DroneTello with optional camera display and mission pads.
        
        Args:
            show_cam (bool): If True, shows live camera feed in a window
            enable_mission_pad (bool): If True, enables mission pad detection
            
        Usage:
            drone = DroneTello()  # Basic connection to default IP
            drone = DroneTello(show_cam=True)  # With camera display
            drone = DroneTello(show_cam=True, enable_mission_pad=True)  # Full features
        """
        super().__init__()

        # Initialize state variables first
        self.show_camera = False
        self._camera_thread = None
        self._stream_active = False
        self._display_active = False
        self.is_land = True  # Drone starts on ground
        self.is_connected = False

        # Try to connect to the Tello drone
        try:
            current_ssid = get_current_ssid()
            if current_ssid:
                print(f"Current Wi-Fi SSID: {current_ssid}", flush=True)
            else:
                print("Current Wi-Fi SSID: unknown", flush=True)
            print(f"Connecting to Tello drone at {getattr(self, 'TELLO_IP', '192.168.10.1')}...", flush=True)
            self.connect()

            # Verify the connection with a real command before treating it as active.
            try:
                battery = self.get_battery()
                print(f"Battery: {battery}%")
            except Exception as e:
                connected_ssid = get_current_ssid()
                print(f"❌ Connection failed: {e}", flush=True)
                if connected_ssid:
                    print(f"Current Wi-Fi SSID: {connected_ssid}", flush=True)
                    if "Tello" not in connected_ssid:
                        print("Your PC appears to be connected to the wrong Wi-Fi network.", flush=True)
                        print("Connect directly to the Tello Wi-Fi and try again.", flush=True)
                    else:
                        print("You are on the Tello Wi-Fi, but the drone is still not responding.", flush=True)
                        print("Try power cycling the Tello and rerunning the script.", flush=True)
                else:
                    print("Could not determine the current Wi-Fi SSID.", flush=True)
                    print("Make sure the Tello is powered on and your computer is connected to its Wi-Fi network.", flush=True)
                self.is_connected = False
                return

            try:
                temperature = self.get_temperature()
                print(f"Temperature: {temperature}°C")
            except Exception as e:
                print(f"Warning: Could not get temperature info: {e}")

            self.is_connected = True
            print("✅ Connected to Tello drone successfully!", flush=True)

            # show camera in realtime if requested
            if show_cam:
                # start video stream
                self._start_video_stream()
                if self._stream_active:
                    self.start_camera_display()

            time.sleep(2)  # Give some time for connection to stabilize

            # enable mission pads if requested
            if enable_mission_pad:
                print("Enabling mission pads...")
                try:
                    self.enable_mission_pads()
                    print("Mission pads enabled successfully")
                except Exception as e:
                    print(f"Warning: Could not enable mission pads: {e}")

            print("Drone Tello initialized successfully.")

        except KeyboardInterrupt:
            print("\n❌ Connection cancelled by user")
            self.is_connected = False
        except Exception as e:
            print(f"❌ Warning: Could not connect to Tello drone: {e}")
            print("Make sure the Tello is powered on and your computer is connected to its Wi-Fi network.")
            self.is_connected = False

    def __del__(self):
        """
        Destructor to ensure cleanup when object is deleted.
        
        Usage: Automatically called when drone object goes out of scope
        """
        try:
            self.cleanup()
        except:
            pass
    
    def _start_video_stream(self):
        """
        Start video stream with error handling and retry mechanism.
        
        Usage: Internal method called automatically when needed
        """
        try:
            print("Starting video stream...")
            self.set_video_resolution(Tello.RESOLUTION_720P)
            self.set_video_bitrate(Tello.BITRATE_5MBPS)
            self.streamon()
            time.sleep(5)  # Wait longer for stream to initialize
            
            # Try multiple times to get frame
            for _ in range(3):
                try:
                    test_frame = self.get_frame_read().frame

                    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    
                    if test_frame is not None:
                        self._stream_active = True
                        print("Video stream started successfully")
                        return
                except Exception as e:
                    print(f"Frame test attempt failed: {e}")
                    time.sleep(1)
                    continue
                    
            print("Warning: Video stream started but no frames available")
            self._stream_active = False
        except Exception as e:
            print(f"Failed to start video stream: {e}")
            self._stream_active = False

    def start_camera_display(self):
        """
        Start displaying camera feed in a GUI window.
        
        Usage:
            drone.start_camera_display()  # Opens camera window
            # Press 'q' in the window to close it
        """
        if not self._stream_active:
            self._start_video_stream()
            
        if self._stream_active:
            # Stop existing camera thread if running
            if self._camera_thread and self._camera_thread.is_alive():
                self.stop_camera_display()
                
            self.show_camera = True
            self._camera_thread = threading.Thread(target=self._camera_loop)
            self._camera_thread.daemon = True
            self._camera_thread.start()
        else:
            print("Cannot start camera display: video stream not active")
        
    def stop_camera_display(self):
        """
        Stop displaying camera feed and close the window.
        
        Usage:
            drone.stop_camera_display()  # Closes camera window
        """
        self.show_camera = False
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2)
        cv2.destroyAllWindows()
        
    def _camera_loop(self):
        """
        Internal method to continuously display camera feed.
        
        Usage: Called automatically by start_camera_display()
        """
        while self.show_camera and self._stream_active:
            try:
                frame = self.get_frame_read().frame

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                
                if frame is not None:
                    # OpenCV expects BGR format, no need to convert
                    cv2.imshow("Tello Camera Feed", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_camera_display()
                        break
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Camera error: {e}")
                self._stream_active = False
                break

    def capture(self, filename="tello_picture.jpg"):
        """
        Capture current frame and save it to pictures/ folder in RGB format.
        
        Args:
            filename (str): Name of the image file to save
            
        Returns:
            str: Full path of saved image file, or None if failed
            
        Usage:
            drone.capture()  # Saves as "tello_picture.jpg"
            drone.capture("my_photo.jpg")  # Saves with custom name
        """
        if not self._stream_active:
            self._start_video_stream()
            
        if not self._stream_active:
            print("Cannot capture: Video stream not available")
            return None
            
        try:
            frame = self.get_frame_read().frame

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            if frame is None:
                print("No frame available for capture")
                return None
                
            path = "pictures/"
            if not os.path.exists(path):
                os.makedirs(path)
                
            full_path = path + filename
            # Save in original BGR format (OpenCV default)
            cv2.imwrite(full_path, frame)
            print(f"Picture saved as {full_path}")
            return full_path
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    # def scan_qr(self, filename):
    #     """
    #     Scan QR code from saved image file and return decoded data.
        
    #     Args:
    #         filename (str): Name of the image file in pictures/ folder
            
    #     Returns:
    #         str: Decoded QR code data, or None if no QR code found
            
    #     Usage:
    #         data = drone.scan_qr("my_photo.jpg")
    #         if data:
    #             print(f"QR code says: {data}")
    #     """
    #     path = "pictures/"
    #     full_path = path + filename
        
    #     if not os.path.exists(full_path):
    #         print(f"File {full_path} not found")
    #         return None
            
    #     try:
    #         frame = cv2.imread(full_path)
            
    #         # Use zxing-cpp to decode QR codes
    #         results = zxingcpp.read_barcodes(frame)

    #         if results:
    #             # Get the first QR code found
    #             qr_data = results[0].text
    #             print(f"QR Code detected in {filename}: {qr_data}")
    #             return qr_data
    #         else:
    #             print(f"No QR code detected in {filename}")
    #             return None
    #     except Exception as e:
    #         print(f"QR scan error: {e}")
    #         return None
        
    def wait(self, seconds):
        """
        Wait for a specified number of seconds with status messages.
        
        Args:
            seconds (int/float): Number of seconds to wait
            
        Usage:
            drone.wait(2)      # Wait 2 seconds
            drone.wait(0.5)    # Wait half a second
        """
        print(f"Waiting for {seconds} seconds...")
        time.sleep(seconds)
        print("Wait complete.")
        
    def takeoff(self):
        """
        Take off and update landing status.
        
        Usage:
            drone.takeoff()  # Drone takes off and is_land becomes False
        """
        super().takeoff()
        self.is_land = False
        
    def land(self):
        """
        Land and update landing status.
        
        Usage:
            drone.land()  # Drone lands and is_land becomes True
        """
        super().land()
        self.is_land = True
        
    def set_wifi_channel(self, channel):
        """
        Set WiFi channel for the drone.
        
        Args:
            channel (int): WiFi channel number
                          2.4GHz channels: 1-14 (most common: 1, 6, 11)
                          5.8GHz channels: 36, 40, 44, 48, 149, 153, 157, 161, 165
        
        Returns:
            str: Response from drone
            
        Usage:
            # Set to 2.4GHz channel
            drone.set_wifi_channel(6)
            
            # Set to 5.8GHz channel  
            drone.set_wifi_channel(165)
        """
        if not hasattr(self, 'is_connected') or not self.is_connected:
            print("❌ Warning: Drone not connected. Cannot set WiFi channel.")
            print(f"Command that would be sent: wifisetchannel {channel}")
            return None
            
        try:
            # Send command mode first to ensure drone is ready
            self.send_command_with_return("command")
            
            # Set the WiFi channel
            response = self.send_command_with_return(f"wifisetchannel {channel}")
            print(f"WiFi channel set to {channel}, Response: {response}")
            return response
        except Exception as e:
            print(f"Error setting WiFi channel: {e}")
            return None
    
    def set_wifi_2_4ghz(self, channel=6):
        """
        Set WiFi to 2.4GHz band with specified channel.
        
        Args:
            channel (int): 2.4GHz channel (1-14, default: 6)
                          Common channels: 1, 6, 11
        
        Returns:
            str: Response from drone
            
        Usage:
            drone.set_wifi_2_4ghz()     # Use default channel 6
            drone.set_wifi_2_4ghz(11)   # Use channel 11
        """
        if channel < 1 or channel > 14:
            print("Warning: 2.4GHz channels should be 1-14")
        
        print(f"Setting WiFi to 2.4GHz band, channel {channel}...")
        return self.set_wifi_channel(channel)
    
    def set_wifi_5_8ghz(self, channel=165):
        """
        Set WiFi to 5.8GHz band with specified channel.
        
        Args:
            channel (int): 5.8GHz channel (36, 40, 44, 48, 149, 153, 157, 161, 165)
                          Default: 165
        
        Returns:
            str: Response from drone
            
        Usage:
            drone.set_wifi_5_8ghz()     # Use default channel 165
            drone.set_wifi_5_8ghz(149)  # Use channel 149
        """
        valid_5ghz_channels = [36, 40, 44, 48, 149, 153, 157, 161, 165]
        if channel not in valid_5ghz_channels:
            print(f"Warning: {channel} may not be a valid 5.8GHz channel")
            print(f"Valid 5.8GHz channels: {valid_5ghz_channels}")
        
        print(f"Setting WiFi to 5.8GHz band, channel {channel}...")
        return self.set_wifi_channel(channel)
    
    def get_wifi_info(self):
        """
        Get WiFi hardware and version information from the drone.
        
        Returns:
            dict: WiFi information including hardware and version
            
        Usage:
            info = drone.get_wifi_info()
            print(f"Hardware: {info['hardware']}")
            print(f"WiFi Version: {info['wifi_version']}")
        """
        if not hasattr(self, 'is_connected') or not self.is_connected:
            print("❌ Warning: Drone not connected. Cannot get WiFi info.")
            print("Commands that would be sent: hardware?, wifiversion?")
            return None
            
        try:
            # Send command mode first
            self.send_command_with_return("command")
            
            # Get hardware info
            hardware_response = self.send_command_with_return("hardware?")
            
            # Get WiFi version
            wifi_version_response = self.send_command_with_return("wifiversion?")
            
            info = {
                'hardware': hardware_response,
                'wifi_version': wifi_version_response
            }
            
            print(f"Hardware: {hardware_response}")
            print(f"WiFi Version: {wifi_version_response}")
            
            return info
        except Exception as e:
            print(f"Error getting WiFi info: {e}")
            return None
            
    def cleanup(self):
        """
        Clean shutdown of drone resources to prevent errors.
        
        Usage:
            drone.cleanup()  # Call before ending program
            # Or use in finally block for automatic cleanup
        """
        try:
            # Stop camera display if it exists
            if hasattr(self, 'show_camera') and self.show_camera:
                self.stop_camera_display()
            
            # Land if drone is still flying (only if connected)
            if hasattr(self, 'is_connected') and self.is_connected:
                if hasattr(self, 'is_land') and not self.is_land:
                    print("Landing drone before cleanup...")
                    try:
                        self.land()
                    except Exception as e:
                        print(f"Warning: Could not land drone: {e}")
            
            # Stop video stream
            if hasattr(self, '_stream_active') and self._stream_active:
                try:
                    self.streamoff()
                    print("Video stream stopped")
                except Exception as e:
                    print(f"Warning: Could not stop video stream: {e}")
            
            # Stop camera thread
            if hasattr(self, '_camera_thread') and self._camera_thread:
                try:
                    self.show_camera = False
                    if self._camera_thread.is_alive():
                        self._camera_thread.join(timeout=1)
                except Exception as e:
                    print(f"Warning: Could not stop camera thread: {e}")
            
            # Close any remaining OpenCV windows
            cv2.destroyAllWindows()
                    
        except Exception as e:
            print(f"Cleanup error: {e}")

