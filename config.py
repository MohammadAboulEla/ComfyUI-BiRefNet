import os
import torch
from folder_paths import models_dir

_class_labels_TR_sorted = "Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht"
class_labels_TR_sorted = _class_labels_TR_sorted.split(", ")

class Config():
    def __init__(self) -> None:
        self.ms_supervision = True
        self.out_ref = True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.mul_scl_ipt = 'cat'
        self.dec_att = 'ASPPDeformable'
        self.squeeze_block = 'BasicDecBlk_x1'
        self.dec_blk = 'BasicDecBlk'
        self.auxiliary_classification = False
        self.freeze_bb = False
        self.batch_size = 2

        self.lat_blk = 'BasicLatBlk'
        self.dec_channels_inter = 'fixed'

        self.bb = 'swin_v1_l'
        self.lateral_channels_in_collection = [3072, 1536, 768, 384]
        self.cxt = [384, 768, 1536]
        
        self.sys_home_dir = models_dir
        self.weights_root_dir = os.path.join(self.sys_home_dir, "BiRefNet")
        # use same weight for all backbones for now
        self.weights = {
            'pvt_v2_b2': os.path.join(self.weights_root_dir, 'model.safetensors'),
            'pvt_v2_b5': os.path.join(self.weights_root_dir, 'model.safetensors'),
            'swin_v1_b': os.path.join(self.weights_root_dir, 'model.safetensors'),
            'swin_v1_l': os.path.join(self.weights_root_dir, 'model.safetensors'),
        }

        self.SDPA_enabled = False
