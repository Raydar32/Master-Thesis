#Per avviare
#  ryu-manager  ryu.app.simple_switch_snort
#  sudo mn --topo tree,1 --mac --controller=remote,ip=127.0.0.1,port=6633 --switch=ovsk,protocols=OpenFlow13
#  sudo snort -i s1-eth1 -A unsock -l /tmp -c /etc/snort/snort.conf


from __future__ import print_function
import array
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib import snortlib
from ryu.lib import hub


class SimpleSwitchSnort(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'snortlib': snortlib.SnortLib}
    def __init__(self, *args, **kwargs):
        super(SimpleSwitchSnort, self).__init__(*args, **kwargs)
        self.snort = kwargs['snortlib']
        self.snort_port = 1
        self.mac_to_port = {}
        socket_config = {'unixsock': True}
        self.snort.set_config(socket_config)
        self.snort.start_socket_server()
        self.monitor_thread = hub.spawn(self._monitor)
        self.datapaths = {}
        self.flowLogFile = "/home/alessandro/flowlog.log"
        self.packetLogFile = "/home/alessandro/packetlog.log"

    @set_ev_cls(ofp_event.EventOFPStateChange,[MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        print("\n datapath",datapath)
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]


    def packet_print(self, pkt):
        #TODO: Trasferire su file
        pkt = packet.Packet(array.array('B', pkt))
        self.logger.info
        eth = pkt.get_protocol(ethernet.ethernet)
        _ipv4 = pkt.get_protocol(ipv4.ipv4)
        _icmp = pkt.get_protocol(icmp.icmp)
        _tcp = pkt.get_protocol(tcp.tcp)


        if _icmp:
            with open(self.packetLogFile, 'a+') as f:
                self.logger.info("%r", _icmp)
                f.write(str(repr(_icmp)+"\n"))
        if _ipv4:
            with open(self.packetLogFile, 'a+') as f:
                self.logger.info("%r", _ipv4)
                f.write(str(repr(_ipv4)+"\n"))
        if eth:
            with open(self.packetLogFile, 'a+') as f:
                self.logger.info("%r", eth)
                f.write(str(repr(eth)+"\n"))
        if _tcp:
            with open(self.packetLogFile, 'a+') as f:
                self.logger.info("%r", _tcp)
                f.write(str(repr(_tcp)+"\n"))

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

        # for p in pkt.protocols:
        #     if hasattr(p, 'protocol_name') is False:
        #         break
        #     print('p: %s' % p.protocol_name)

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        #datapath, in-port, eth-dst, out-port, packets bytes
        #TODO: Trasferire su file
        for stat in sorted([flow for flow in body if flow.priority == 1],key=lambda flow: (flow.match['in_port'],flow.match['eth_dst'])):
            self.logger.info('%016x %8x %17s %8x %8d %8d',
            ev.msg.datapath.id,
            stat.match['in_port'], stat.match['eth_dst'],
            stat.instructions[0].actions[0].port,
            stat.packet_count, stat.byte_count)

            with open(self.flowLogFile, 'a+') as f:
                f.write(str(ev.msg.datapath.id) + " " + str( stat.match['in_port'])  + " " + str(stat.match['eth_dst'])  + " " + str(stat.instructions[0].actions[0].port)  + " " +  str(stat.packet_count)  + " " + str(stat.byte_count) + "\n")

    @set_ev_cls(snortlib.EventAlert, MAIN_DISPATCHER)
    def _dump_alert(self, ev):
        msg = ev.msg
        self.packet_print(msg.pkt)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)



    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port),
                   parser.OFPActionOutput(self.snort_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
