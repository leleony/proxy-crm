QT.network_private.VERSION = 5.15.8
QT.network_private.name = QtNetwork
QT.network_private.module =
QT.network_private.libs = $$QT_MODULE_LIB_BASE
QT.network_private.includes = $$QT_MODULE_INCLUDE_BASE/QtNetwork/5.15.8 $$QT_MODULE_INCLUDE_BASE/QtNetwork/5.15.8/QtNetwork
QT.network_private.frameworks =
QT.network_private.depends = core_private network
QT.network_private.uses = openssl
QT.network_private.module_config = v2 internal_module
QT.network_private.enabled_features = openssl-linked openssl netlistmgr system-proxies
QT.network_private.disabled_features = libproxy linux-netlink
QMAKE_LIBS_OPENSSL = -llibssl -llibcrypto -lUser32 -lWs2_32 -lAdvapi32 -lCrypt32
QMAKE_LIBS_NETWORK = ws2_32.lib
