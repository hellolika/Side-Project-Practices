CREATE PROCEDURE [dbo].[HRMS_GetAllReClockRecords.1.0.0]
	@memberId INT
AS
BEGIN
	SET NOCOUNT ON;

	SELECT [RequestId], [Date], [Time], [IsClockIn], [Status] FROM [dbo].[ReClockRecord] WITH(NOLOCK)
	WHERE [MemberId] = @memberId

END