//simulator bytom miner
package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net"
	"os"
	"runtime"
	"strconv"
	"time"

	//"github.com/bytom/consensus/difficulty"
	"github.com/bytom/mining/tensority"
	"github.com/bytom/protocol/bc"
	"github.com/bytom/protocol/bc/types"
	bytomutil "github.com/bytom/util"
)

type RpcRequest struct {
	Method string      `json:"method"`
	Params interface{} `json:"params"`
	ID     string      `json:"id"`
	Worker string      `json:"worker"`
}

type RpcResponse struct {
	ID         string          `json:"id"`
	Result     json.RawMessage `json:"result"`
	Error      json.RawMessage `json:"error"`
	RpcVersion string          `json:"jsonrpc"`
}

type Miner struct {
	ID          string
	Pool        string
	Status      bool
	Address     string
	LatestJobId string
	MsgId       uint64
	Session     net.Conn
	//dataCh      chan string
	QuitCh chan struct{}
}

type PoolErr struct {
	Code    uint64 `json:"code"`
	Message string `json:"message"`
}

type MineJob struct {
	Version    string `json:"version"`
	Height     string `json:"height"`
	PreBlckHsh string `json:"previous_block_hash"`
	Timestamp  string `json:"timestamp"`
	TxMkRt     string `json:"transactions_merkle_root"`
	TxSt       string `json:"transaction_status_hash"`
	Nonce      string `json:"nonce"`
	Bits       string `json:"bits"`
	JobId      string `json:"job_id"`
	Seed       string `json:"seed"`
	Target     string `json:"target"`
}

type Result struct {
	Id     string  `json:"id"`
	Job    MineJob `json:"job"`
	Status string  `json:"status"`
}

type StratumResp struct {
	Id      string  `json:"id"`
	Jsonrpc string  `json:"jsonrpc, omitempty"`
	Result  Result  `json:"result, omitempty"`
	Error   PoolErr `json:"error, omitempty"`
}

type MineJobntf struct {
	Jsonrpc string  `json:"jsonrpc, omitempty"`
	Method  string  `json:"method, omitempty"`
	Params  MineJob `json:"params, omitempty"`
}

type SubmitReq struct {
	Id    string `json:"id"`
	JobId string `json:"job_id"`
	Nonce string `json:"nonce"`
}

type LoginReq struct {
	Login    string `json:"login"`
	Password string `json:"pass"`
	Agent    string `json:"agent"`
}

type SubmitWorkReq struct {
	BlockHeader *types.BlockHeader `json:"block_header"`
}

var (
	poolAddr string
	login    string
	DEBUG    bool = false
	maxNonce      = ^uint64(0) // 2^64 - 1 = 18446744073709551615
	Diff1         = StringToBig("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
)

func main() {
	pool := flag.String("pool", "stratum.btcc.com:9221", "Mining Pool stratum+tcp:// Addr")
	user := flag.String("user", "bm1qh7e8309j24faltn5auurwqawlza5ylwxkzudsc.1", "login user , bytomAddress.[rigname]")
	thread := flag.Int("thread", 2, "runtime max thread")
	flag.Parse()
	poolAddr = *pool
	login = *user
	os.Setenv("BYTOM_URL", "127.0.0.1:9888")
	if *thread > 1 {
		runtime.GOMAXPROCS(*thread)
	}
reboot:
	done := make(chan struct{})
	log.Printf("Running with %v threads", *thread)
	startMining(done)
	select {
	case <-done:
		goto reboot
		log.Printf("Miner test finished")
	}
}

//start mine bytom
func startMining(closeCh chan struct{}) error {
	log.Println("Miner  start")

	go func(done chan struct{}) {
		miner, err := NewMiner(login, poolAddr)
		if err != nil {
			close(done)
			return
		}
		miner.Start()
		close(done)
	}(closeCh)

	return nil
}

func NewMiner(login, pool string) (m *Miner, err error) {
	/*conn, err := net.Dial("tcp", pool)
	if err != nil {
		return
	}*/

	m = &Miner{
		ID:      login,
		Address: login, //the address to receive miner profit
		Pool:    pool,
		//Session:     conn,
		Status:      true,
		LatestJobId: "",
		MsgId:       0,
		//dataCh:      make(chan string, 64),
		QuitCh: make(chan struct{}),
	}

	return
}

func (m *Miner) Login() (err error) {
	conn, err := net.Dial("tcp", m.Pool)
	if err != nil {
		return
	}
	m.Session = conn
	m.Status = true
	req := RpcRequest{
		ID:     m.ID,
		Method: "login",
		Params: LoginReq{Login: m.Address, Password: "password", Agent: "bmminer/2.0.0"},
		Worker: m.ID,
	}

	if err := m.WriteStratumRequest(req, time.Now().Add(10*time.Second)); err != nil {
		log.Println("error in test miner login()")
		log.Println(err.Error())
		return err
	}
	reply, err := bufio.NewReader(m.Session).ReadBytes('\n')
	if len(reply) == 0 || err != nil {
		close(m.QuitCh)
	}
	var resp StratumResp
	err = json.Unmarshal(reply, &resp)
	if err != nil {
		return
	}
	m.LatestJobId = resp.Result.Job.JobId
	go func(job MineJob) {
		m.Mine(job)
	}(resp.Result.Job)
	return nil
}

func (m *Miner) Start() error {

	//subscribe server login
	if err := m.Login(); err != nil {
		log.Println(err.Error())
		close(m.QuitCh)
		//return err
	}
	/*reply, err := bufio.NewReader(m.Session).ReadBytes('\n')
	if len(reply) == 0 || err != nil {
		close(m.QuitCh)
	}
	var resp StratumResp
	json.Unmarshal(reply, &resp)
	m.LatestJobId = resp.Result.Job.JobId
	go func(job MineJob) {
		m.Mine(job)
	}(resp.Result.Job)*/
	//listen to the server message
	go func() {
		for {
			message, err := bufio.NewReader(m.Session).ReadBytes('\n')
			if len(message) == 0 || err != nil {
				close(m.QuitCh)
				break
			}

			fmt.Println("Message from server: ", string(message))
			var jobntf MineJobntf
			json.Unmarshal(message, &jobntf)

			if jobntf.Method == "job" {
				log.Printf("----new job received----\n%s\n", message)
				m.LatestJobId = jobntf.Params.JobId
				go func(job MineJob) {
					m.Mine(job)
				}(jobntf.Params)
			} else {
				log.Printf("Received: %s\n", message)
			}
		}
	}()

	select {
	case <-m.QuitCh:
		log.Println("Miner ", m.ID, m.Address, "quit")
		//close(m.dataCh)
		m.Session.Close()
		m.Status = false
	}
	return nil
}

//send stratum request to mining pool
func (m *Miner) WriteStratumRequest(req RpcRequest, deadline time.Time) error {

	data, err := json.Marshal(req)
	if err != nil {
		log.Println("WriteStratumRequest marshal", err.Error())
		return err
	}

	data = append(data, byte('\n'))

	if err := m.Session.SetWriteDeadline(deadline); err != nil {
		return err
	}

	if _, err := m.Session.Write(data); err != nil {
		return err
	}

	return nil
}

func (m *Miner) Mine(job MineJob) bool {
	seedHash, err1 := DecodeHash(job.Seed)
	PreBlckHsh, err2 := DecodeHash(job.PreBlckHsh)
	TxMkRt, err3 := DecodeHash(job.TxMkRt)
	TxSt, err4 := DecodeHash(job.TxSt)
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		return false
	}

	bh := &types.BlockHeader{
		Version:           str2ui64Bg(job.Version),
		Height:            str2ui64Bg(job.Height),
		PreviousBlockHash: PreBlckHsh,
		Timestamp:         str2ui64Bg(job.Timestamp),
		Bits:              str2ui64Bg(job.Bits),
		BlockCommitment: types.BlockCommitment{
			TransactionsMerkleRoot: TxMkRt,
			TransactionStatusHash:  TxSt,
		},
	}
	if DEBUG {
		viewParsing(bh, job)
	}

	log.Printf("Job %s: Mining at height: %d\n", job.JobId, bh.Height)

	log.Printf("Job %s: Old target: %v\n", job.JobId, CompactToBig(bh.Bits))
	newDiff := getNewTargetDiff(job.Target)
	log.Printf("Job %s: New target: %v\n", job.JobId, newDiff)

	nonce := str2ui64Li(job.Nonce)
	log.Printf("Job %s: Start from nonce:\t0x%016x = %d\n", job.JobId, nonce, nonce)

	for i := nonce; i <= maxNonce; i++ {
		if job.JobId != m.LatestJobId || !m.Status {
			log.Printf("Job %s: Expired", job.JobId)
			return false
		} else {
			// log.Printf("Checking PoW with nonce: 0x%016x = %d\n", i, i)
			bh.Nonce = i
			headerHash := bh.Hash()
			if DEBUG {
				fmt.Printf("Job %s: HeaderHash: %v\n", job.JobId, headerHash.String())
			}

			// if difficulty.CheckProofOfWork(&headerHash, &seedHash, bh.Bits) {
			if t := CheckPOW(&headerHash, &seedHash, BigToCompact(newDiff), bh.Bits); t > 0 {
				log.Printf("Job %s: Target found! Proof hash: 0x%v\n", job.JobId, headerHash.String())

				if t == 2 {
					log.Println("Block found!")
					go func(blockheader types.BlockHeader) {
						m.SubmitBlock(blockheader)
					}(*bh)
				} else if t == 1 {
					go func(jobid string, i uint64) {
						m.SubmitWork(jobid, i)
					}(job.JobId, i)
				}
				//m.SubmitWork(job.JobId, i)
			}
		}
	}
	log.Printf("Job %s: Stop at nonce:\t\t0x%016x = %d\n", job.JobId, bh.Nonce, bh.Nonce)
	return false
}

//{"id": "antminer_1", "job_id": "1285153", "nonce": "0000026f80000ab9"}
func (m *Miner) SubmitWork(jobId string, nonce uint64) (err error) {
	req := RpcRequest{
		ID:     m.ID,
		Method: "submit",
		Params: SubmitReq{
			Id:    m.ID,
			JobId: jobId,
			Nonce: getNonceStr(nonce),
		},
		Worker: m.ID,
	}

	if err := m.WriteStratumRequest(req, time.Now().Add(10*time.Second)); err != nil {
		log.Println("error in test miner submitwork()")
		log.Println(err.Error())
		return err
	}
	return nil
}

func (m *Miner) SubmitBlock(bh types.BlockHeader) {
	_, success := bytomutil.ClientCall("/submit-work", &SubmitWorkReq{BlockHeader: &bh})
	if success == 0 {
		log.Println("Mined new Block!")
	}
}

func viewParsing(bh *types.BlockHeader, job MineJob) {
	log.Println("Printing parsing result:")
	fmt.Println("\tVersion:", bh.Version)
	fmt.Println("\tHeight:", bh.Height)
	fmt.Println("\tPreviousBlockHash:", bh.PreviousBlockHash.String())
	fmt.Println("\tTimestamp:", bh.Timestamp)
	fmt.Println("\tbits_str:", job.Bits)
	fmt.Println("\tBits:", bh.Bits)
	fmt.Println("\tTransactionsMerkleRoot:", bh.BlockCommitment.TransactionsMerkleRoot.String())
	fmt.Println("\tTransactionStatusHash:", bh.BlockCommitment.TransactionStatusHash.String())
	fmt.Println("\ttarget_str:", job.Target)
	fmt.Println("\ttarget_ui64Bg:", str2ui64Bg(job.Target))
}

func str2ui64Bg(str string) uint64 {
	ui64, _ := strconv.ParseUint(strSwitchEndian(str), 16, 64)
	return ui64
}

func str2ui64Li(str string) uint64 {
	ui64, _ := strconv.ParseUint(str, 16, 64)
	return ui64
}

func strSwitchEndian(oldstr string) string {

	slen := len(oldstr)
	if slen%2 != 0 {
		panic("hex string format error")
	}

	newstr := ""
	for i := 0; i < slen; i += 2 {
		newstr += oldstr[slen-i-2 : slen-i]
	}

	return newstr
}

func StringToBig(h string) *big.Int {
	n := new(big.Int)
	n.SetString(h, 0)
	return n
}

func reverse(src []byte) []byte {
	dst := make([]byte, len(src))
	for i := len(src); i > 0; i-- {
		dst[len(src)-i] = src[i-1]
	}
	return dst
}

func DecodeHash(s string) (h bc.Hash, err error) {
	err = h.UnmarshalText([]byte(s))
	return h, err
}

func getNewTargetDiff(target string) *big.Int {
	padded := make([]byte, 32)
	targetHex := target
	decoded, _ := hex.DecodeString(targetHex)
	decoded = reverse(decoded)
	copy(padded[:len(decoded)], decoded)
	newDiff := new(big.Int).SetBytes(padded)
	//newDiff = new(big.Int).Div(Diff1, newDiff)
	//log.Printf(" Old target: %v\n", difficulty.CompactToBig(blockheaderBits))
	//newDiff = new(big.Int).Mul(difficulty.CompactToBig(blockheaderBits), newDiff)
	//log.Printf("New target: %v\n", newDiff)
	return newDiff
}

func getNonceStr(i uint64) string {
	nonceStr := strconv.FormatUint(i, 16)
	nonceStr = fmt.Sprintf("%016s", nonceStr)
	return nonceStr
}

func HashToBig(hash *bc.Hash) *big.Int {
	// reverse the bytes of the hash (little-endian) to use it in the big
	// package (big-endian)
	buf := hash.Byte32()
	blen := len(buf)
	for i := 0; i < blen/2; i++ {
		buf[i], buf[blen-1-i] = buf[blen-1-i], buf[i]
	}

	return new(big.Int).SetBytes(buf[:])
}

//0 not pass 1 pass target not pass bits,2 pass bits network is new block
func CheckPOW(hash, seed *bc.Hash, target, bits uint64) (ret int) {
	compareHash := tensority.AIHash.Hash(hash, seed)
	tmp := HashToBig(compareHash)

	if tmp.Cmp(CompactToBig(target)) <= 0 {
		ret = 1
	} else {
		ret = 0
		return
	}
	if tmp.Cmp(CompactToBig(bits)) <= 0 {
		ret = 2
	}
	return
}

func CompactToBig(compact uint64) *big.Int {
	// Extract the mantissa, sign bit, and exponent.
	mantissa := compact & 0x007fffffffffffff
	isNegative := compact&0x0080000000000000 != 0
	exponent := uint(compact >> 56)

	var bn *big.Int
	if exponent <= 3 {
		mantissa >>= 8 * (3 - exponent)
		bn = big.NewInt(int64(mantissa))
	} else {
		bn = big.NewInt(int64(mantissa))
		bn.Lsh(bn, 8*(exponent-3))
	}

	if isNegative {
		bn = bn.Neg(bn)
	}

	return bn
}

func BigToCompact(n *big.Int) uint64 {
	if n.Sign() == 0 {
		return 0
	}

	var mantissa uint64
	// Bytes() returns the absolute value of n as a big-endian byte slice
	exponent := uint(len(n.Bytes()))

	// Bits() returns the absolute value of n as a little-endian uint64 slice
	if exponent <= 3 {
		mantissa = uint64(n.Bits()[0])
		mantissa <<= 8 * (3 - exponent)
	} else {
		tn := new(big.Int).Set(n)
		// Since the base for the exponent is 256, the exponent can be treated
		// as the number of bytes to represent the full 256-bit number. And as
		// the exponent is treated as the number of bytes, Rsh 8*(exponent-3)
		// makes sure that the shifted tn won't occupy more than 8*3=24 bits,
		// and can be read from Bits()[0], which is 64-bit
		mantissa = uint64(tn.Rsh(tn, 8*(exponent-3)).Bits()[0])
	}

	if mantissa&0x0080000000000000 != 0 {
		mantissa >>= 8
		exponent++
	}

	compact := uint64(exponent)<<56 | mantissa
	if n.Sign() < 0 {
		compact |= 0x0080000000000000
	}
	return compact
}
