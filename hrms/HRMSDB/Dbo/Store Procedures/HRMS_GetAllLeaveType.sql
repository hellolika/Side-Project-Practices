CREATE PROCEDURE [dbo].[HRMS_GetAllLeaveType.2.0.0]

AS
BEGIN
	SET NOCOUNT ON;

	SELECT [TypeId], [Type], [DefaultLeavesGranted], [IsEnable], [IsLimited] FROM [dbo].[LeaveType] WITH(NOLOCK)

END