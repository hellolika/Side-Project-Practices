using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class GetMontlyTransferResponse
{
    [JsonProperty("Total")]
    public double Total { get; set; }

    [JsonProperty("TransferList")]
    public List<TransferResponse> TransferListResponse { get; set; }
}